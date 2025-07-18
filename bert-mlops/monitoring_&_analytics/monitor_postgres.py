# monitor_postgres.py - Class-based architecture
import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Optional, Any, Dict, List, Union
import psycopg2
from psycopg2.extensions import connection as Connection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import psycopg2

POSTGRES_AVAILABLE = True

PLOTLY_AVAILABLE = True
MATPLOTLIB_AVAILABLE = True

warnings.filterwarnings('ignore')



class DatabaseManager:
    """Handles PostgreSQL database connections and queries"""

    def __init__(self, db_config: Dict[str, Union[str, int]]):
        self.db_config = db_config
        self.postgres_available = POSTGRES_AVAILABLE

    def get_connection(self) -> Connection:
        """Get PostgreSQL connection"""
        if not self.postgres_available:
            raise RuntimeError("PostgreSQL connection not available")

        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.OperationalError as e:
            print(f"❌ Database connection failed: {e}")
            print("🔧 Make sure PostgreSQL is running and MLflow database exists")
            raise
        except Exception as e:
            print(f"❌ Database error: {e}")
            raise

    def test_connection(self) -> bool:
        """Test database connection"""
        if not self.postgres_available:
            print("❌ PostgreSQL client not available")
            return False

        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result:
                    print("✅ PostgreSQL connection successful!")
                    return True
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute a query and return DataFrame"""
        if not self.postgres_available:
            print("❌ PostgreSQL not available for querying")
            return pd.DataFrame()

        conn = None
        try:
            conn = self.get_connection()
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()


class MLflowDataProvider:
    """Provides data from MLflow database"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def get_experiments(self) -> pd.DataFrame:
        """Query all experiments from PostgreSQL"""
        query = """
        SELECT 
            experiment_id,
            name,
            artifact_location,
            lifecycle_stage,
            creation_time,
            last_update_time
        FROM experiments 
        WHERE lifecycle_stage = 'active'
        ORDER BY creation_time DESC;
        """

        df = self.db_manager.execute_query(query)

        if not df.empty:
            # Proper timestamp conversion with error handling
            df['creation_time'] = pd.to_datetime(df['creation_time'], unit='ms', errors='coerce')
            df['last_update_time'] = pd.to_datetime(df['last_update_time'], unit='ms', errors='coerce')

        return df

    def get_runs_summary(self, experiment_id: Optional[str] = None, days_back: int = 30) -> pd.DataFrame:
        """Query runs with metrics"""
        base_query = """
        SELECT 
            r.run_uuid,
            r.experiment_id,
            r.name as run_name,
            r.user_id,
            r.status,
            r.start_time,
            r.end_time,
            r.lifecycle_stage
        FROM runs r
        WHERE r.lifecycle_stage = 'active'
        """

        params: List[Any] = []

        if experiment_id:
            base_query += " AND r.experiment_id = %s"
            params.append(experiment_id)

        if days_back > 0:
            cutoff_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            base_query += " AND r.start_time >= %s"
            params.append(cutoff_time)

        base_query += " ORDER BY r.start_time DESC;"

        runs_df = self.db_manager.execute_query(base_query, params if params else None)

        if not runs_df.empty:
            # Proper timestamp conversion
            runs_df['start_time'] = pd.to_datetime(runs_df['start_time'], unit='ms', errors='coerce')
            runs_df['end_time'] = pd.to_datetime(runs_df['end_time'], unit='ms', errors='coerce')

            # Calculate duration with proper null handling
            runs_df['duration_minutes'] = (
                                                  runs_df['end_time'] - runs_df['start_time']
                                          ).dt.total_seconds() / 60

        return runs_df

    def get_run_metrics(self, run_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Query metrics for runs"""
        base_query = """
        SELECT 
            run_uuid,
            key as metric_name,
            value as metric_value,
            timestamp,
            step,
            is_nan
        FROM metrics
        """

        params: List[str] = []

        if run_ids:
            placeholders = ','.join(['%s'] * len(run_ids))
            base_query += f" WHERE run_uuid IN ({placeholders})"
            params.extend(run_ids)

        base_query += " ORDER BY run_uuid, key, step;"

        df = self.db_manager.execute_query(base_query, params if params else None)

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

        return df


class APIClient:
    """Handles API interactions"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def check_health(self) -> bool:
        """Check API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ API is healthy")
                print(f"   Model loaded: {health_data.get('model_loaded', False)}")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ API health check error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during health check: {e}")
            return False

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get API metrics"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API metrics failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ API metrics error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error getting metrics: {e}")
            return None

    def simulate_traffic(self, num_requests: int = 10) -> List[Dict[str, Any]]:
        """Simulate API traffic"""
        test_texts = [
            "I absolutely love this product! Amazing quality!",
            "Great service, highly recommend to everyone",
            "This exceeded my expectations completely",
            "This product is terrible, complete waste of money",
            "Worst service I've ever experienced",
            "Product broke immediately after opening",
            "The product is okay, nothing special",
            "Average quality for the price"
        ]

        print(f"🚀 Simulating {num_requests} API requests...")

        results: List[Dict[str, Any]] = []

        for i in range(num_requests):
            text = np.random.choice(test_texts)

            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={"text": text, "return_confidence": True},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    results.append(result)
                    confidence = result.get('confidence', 0)
                    sentiment = result.get('sentiment', 'unknown')
                    print(f"Request {i + 1}: {sentiment} ({confidence:.3f})")
                else:
                    print(f"Request {i + 1}: Failed with status {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Request {i + 1}: Request error - {str(e)}")
            except Exception as e:
                print(f"Request {i + 1}: Unexpected error - {str(e)}")

            time.sleep(0.2)  # Small delay

        print(f"✅ Completed {len(results)}/{num_requests} requests")
        return results


class DashboardCreator:
    """Creates different types of dashboards"""

    def __init__(self, data_provider: MLflowDataProvider):
        self.data_provider = data_provider

    def create_simple_dashboard(self) -> bool:
        """Create a simple text-based dashboard"""
        print("📊 Creating monitoring dashboard...")

        # Get data
        experiments_df = self.data_provider.get_experiments()
        runs_df = self.data_provider.get_runs_summary(days_back=30)

        print("\n" + "=" * 50)
        print("🎯 MLFLOW MONITORING DASHBOARD")
        print("=" * 50)

        # Experiments section
        print(f"\n📊 EXPERIMENTS")
        print(f"Total experiments: {len(experiments_df)}")
        if not experiments_df.empty:
            for _, exp in experiments_df.head(3).iterrows():
                print(f"  • {exp['name']} (ID: {exp['experiment_id']})")

        # Runs section
        print(f"\n📈 RUNS (Last 30 days)")
        print(f"Total runs: {len(runs_df)}")
        if not runs_df.empty:
            successful_runs = runs_df[runs_df['status'] == 'FINISHED']
            failed_runs = runs_df[runs_df['status'] == 'FAILED']
            print(f"  • Successful: {len(successful_runs)}")
            print(f"  • Failed: {len(failed_runs)}")

            # Safe duration calculation
            valid_durations = runs_df['duration_minutes'].dropna()
            if len(valid_durations) > 0:
                duration_mean = valid_durations.mean()
                print(f"  • Average duration: {duration_mean:.2f} minutes")
            else:
                print(f"  • Average duration: N/A")

        # Metrics section
        if not runs_df.empty:
            run_ids = runs_df['run_uuid'].tolist()[:10]
            metrics_df = self.data_provider.get_run_metrics(run_ids)

            if not metrics_df.empty:
                print(f"\n📊 METRICS")
                unique_metrics = metrics_df['metric_name'].unique()
                print(f"Available metrics: {', '.join(unique_metrics)}")

                # Show latest accuracy if available
                accuracy_metrics = metrics_df[
                    metrics_df['metric_name'].str.contains('accuracy', case=False, na=False)
                ]
                if not accuracy_metrics.empty:
                    latest_accuracy = accuracy_metrics.iloc[-1]['metric_value']
                    print(f"  • Latest accuracy: {latest_accuracy:.4f}")

        return True

    def create_plotly_dashboard(self) -> bool:
        """Create interactive Plotly dashboard"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly not available - using simple dashboard")
            return self.create_simple_dashboard()

        print("📊 Creating interactive Plotly dashboard...")

        # Get data
        experiments_df = self.data_provider.get_experiments()
        runs_df = self.data_provider.get_runs_summary(days_back=30)

        if runs_df.empty:
            print("❌ No runs found. Train a model first!")
            return False

        # Get metrics
        run_ids = runs_df['run_uuid'].tolist()[:20]
        metrics_df = self.data_provider.get_run_metrics(run_ids)

        # Create subplots with proper error handling
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Training Progress Over Time',
                    'Model Performance Distribution',
                    'Training Duration Analysis',
                    'Run Success Rate'
                ],
                specs=[
                    [{"secondary_y": True}, {"type": "histogram"}],
                    [{"type": "scatter"}, {"type": "pie"}]
                ]
            )
        except Exception as e:
            print(f"❌ Error creating subplots: {e}")
            return False

        # 1. Training Progress
        if not metrics_df.empty:
            accuracy_data = metrics_df[
                metrics_df['metric_name'].str.contains('accuracy', case=False, na=False)
            ]
            if not accuracy_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accuracy_data['timestamp'],
                        y=accuracy_data['metric_value'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )

        # 2. Performance Distribution
        if not metrics_df.empty:
            f1_data = metrics_df[metrics_df['metric_name'].str.contains('f1', case=False, na=False)]
            if not f1_data.empty:
                fig.add_trace(
                    go.Histogram(
                        x=f1_data['metric_value'],
                        name='F1 Score Distribution',
                        nbinsx=20
                    ),
                    row=1, col=2
                )

        # 3. Training Duration - proper null handling
        duration_data = runs_df.dropna(subset=['duration_minutes'])
        if not duration_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=duration_data['start_time'],
                    y=duration_data['duration_minutes'],
                    mode='markers',
                    name='Training Duration',
                    marker=dict(
                        size=8,
                        color=duration_data['duration_minutes'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=2, col=1
            )

        # 4. Run Success Rate
        status_counts = runs_df['status'].value_counts()
        if not status_counts.empty:
            fig.add_trace(
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    name='Run Status'
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="MLflow Monitoring Dashboard - PostgreSQL Backend",
            showlegend=True
        )

        # Save and show with error handling
        try:
            fig.write_html("mlflow_monitoring_dashboard.html")
            fig.show()
            print("✅ Dashboard saved as 'mlflow_monitoring_dashboard.html'")
            return True
        except Exception as e:
            print(f"❌ Dashboard creation failed: {e}")
            return False


class DependencyChecker:
    """Checks and reports on missing dependencies"""

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check which dependencies are available"""
        return {
            'postgres': POSTGRES_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE,
            'matplotlib': MATPLOTLIB_AVAILABLE
        }

    @staticmethod
    def report_missing_dependencies():
        """Report missing dependencies"""
        deps = DependencyChecker.check_dependencies()
        missing_deps = []

        if not deps['postgres']:
            missing_deps.append("psycopg2-binary")
        if not deps['plotly']:
            missing_deps.append("plotly")
        if not deps['matplotlib']:
            missing_deps.append("matplotlib seaborn")

        if missing_deps:
            print(f"💡 Install missing dependencies:")
            for dep in missing_deps:
                print(f"   pip install {dep}")

        return missing_deps


class PostgresMLflowMonitor:
    """Main monitoring class that coordinates all components"""

    def __init__(self,
                 db_host: str = "localhost",
                 db_port: int = 5432,
                 db_name: str = "mlflow_db",
                 db_user: str = "mlflow_user",
                 db_password: str = "pass",
                 api_url: str = "http://localhost:8000"):

        # Database configuration
        db_config = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }

        # Initialize components
        self.db_manager = DatabaseManager(db_config)
        self.data_provider = MLflowDataProvider(self.db_manager)
        self.api_client = APIClient(api_url)
        self.dashboard_creator = DashboardCreator(self.data_provider)
        self.dependency_checker = DependencyChecker()

    def run_full_monitoring(self) -> None:
        """Run complete monitoring workflow"""
        print("🚀 Enhanced MLflow Monitoring with PostgreSQL")
        print("=" * 50)

        # Test connections
        print("🔍 Testing connections...")

        # Test database
        db_connected = self.db_manager.test_connection()

        # Test API
        api_healthy = self.api_client.check_health()

        # Handle database monitoring
        if db_connected:
            self._handle_database_monitoring()
        else:
            print("❌ Database connection failed. Make sure PostgreSQL is running.")

        # Handle API monitoring
        if api_healthy:
            self._handle_api_monitoring()
        else:
            print("❌ API not available. Start the API server first.")

        print(f"\n✅ Monitoring check complete!")

        # Show dependency status
        self.dependency_checker.report_missing_dependencies()

    def _handle_database_monitoring(self):
        """Handle database-related monitoring"""
        experiments = self.data_provider.get_experiments()
        print(f"\n📊 Found {len(experiments)} experiments")

        runs = self.data_provider.get_runs_summary(days_back=30)
        print(f"📈 Found {len(runs)} runs in last 30 days")

        if not runs.empty:
            # Create dashboard
            if PLOTLY_AVAILABLE:
                self.dashboard_creator.create_plotly_dashboard()
            else:
                self.dashboard_creator.create_simple_dashboard()
        else:
            print("❌ No runs found. Train a model first with training script")
            self.dashboard_creator.create_simple_dashboard()

    def _handle_api_monitoring(self):
        """Handle API-related monitoring"""
        # Simulate some traffic
        print("\n🚀 Testing API with sample requests...")
        self.api_client.simulate_traffic(num_requests=5)

        # Get API metrics
        api_metrics = self.api_client.get_metrics()
        if api_metrics:
            print(f"\n📊 API Metrics:")
            for key, value in api_metrics.items():
                print(f"  {key}: {value}")

    # Convenience methods for individual operations
    def test_database_connection(self) -> bool:
        """Test database connection"""
        return self.db_manager.test_connection()

    def test_api_health(self) -> bool:
        """Test API health"""
        return self.api_client.check_health()

    def create_dashboard(self, use_plotly: bool = True) -> bool:
        """Create dashboard"""
        if use_plotly and PLOTLY_AVAILABLE:
            return self.dashboard_creator.create_plotly_dashboard()
        else:
            return self.dashboard_creator.create_simple_dashboard()

    def simulate_api_requests(self, num_requests: int = 10) -> List[Dict[str, Any]]:
        """Simulate API requests"""
        return self.api_client.simulate_traffic(num_requests)


def main() -> None:
    """Main function"""
    monitor = PostgresMLflowMonitor()
    monitor.run_full_monitoring()


if __name__ == "__main__":
    main()