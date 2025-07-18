# serve_model.py - Class-based architecture
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
import torch
import numpy as np
import time
import logging
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    text: str
    return_confidence: bool = True


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    prediction_time_ms: float
    model_version: str
    timestamp: str


class ModelMonitor:
    """Monitoring class for tracking predictions"""

    def __init__(self):
        self.predictions = []
        self.start_time = datetime.now()

    def log_prediction(self, text: str, sentiment: str, confidence: float, latency: float, model_version: str):
        prediction_log = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'sentiment': sentiment,
            'confidence': confidence,
            'latency_ms': latency,
            'model_version': model_version
        }
        self.predictions.append(prediction_log)

        # Keep only last 1000 predictions
        if len(self.predictions) > 1000:
            self.predictions = self.predictions[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        if not self.predictions:
            return {"message": "No predictions yet"}

        recent_predictions = self.predictions[-100:]
        if not recent_predictions:
            return {"message": "No recent predictions"}

        avg_latency = np.mean([p['latency_ms'] for p in recent_predictions])
        avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
        sentiment_dist = {
            'positive': len([p for p in recent_predictions if p['sentiment'] == 'positive']),
            'negative': len([p for p in recent_predictions if p['sentiment'] == 'negative'])
        }

        return {
            'total_predictions': len(self.predictions),
            'recent_predictions': len(recent_predictions),
            'avg_latency_ms': round(avg_latency, 2),
            'avg_confidence': round(avg_confidence, 3),
            'sentiment_distribution': sentiment_dist,
            'uptime_hours': round((datetime.now() - self.start_time).total_seconds() / 3600, 2)
        }


class ModelManager:
    """Handles model loading and prediction logic"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_info = {}
        self.is_loaded = False

    def configure_mlflow(self):
        """Configure MLflow connection"""
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        # Configure S3/MinIO if available
        if os.getenv("AWS_ACCESS_KEY_ID"):
            os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio_user")
            os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_password")
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

        logger.info(f"MLflow configured with URI: {mlflow_uri}")

    def load_from_mlflow(self, run_id: Optional[str] = None, model_version: Optional[str] = None) -> bool:
        """Load model from MLflow"""
        try:
            self.configure_mlflow()

            # Determine model URI
            model_uri = self._get_model_uri(run_id, model_version)
            if not model_uri:
                logger.error("Could not determine model URI")
                return False

            logger.info(f"Loading model from: {model_uri}")

            # Load model
            loaded_model = mlflow.pytorch.load_model(model_uri)
            if loaded_model is None:
                logger.error("MLflow returned None for model")
                return False

            self.model = loaded_model
            self.model.eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            # Set model info
            self.model_info = {
                'version': model_version or f"run-{(run_id or 'latest')[:8]}",
                'model_uri': model_uri,
                'source': 'mlflow'
            }

            self.is_loaded = True
            logger.info("Model loaded successfully from MLflow")
            return True

        except Exception as e:
            logger.error(f"Failed to load from MLflow: {e}")
            return False

    def _get_model_uri(self, run_id: Optional[str], model_version: Optional[str]) -> Optional[str]:
        """Get model URI based on run_id or model_version"""
        if run_id:
            return f"runs:/{run_id}/model"

        if model_version:
            return f"models:/bert-sentiment-classifier/{model_version}"

        # Try to find latest model
        try:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions("bert-sentiment-classifier")

            if latest_versions:
                return f"models:/bert-sentiment-classifier/{latest_versions[0].version}"

            # Fallback to latest experiment run
            experiment = mlflow.get_experiment_by_name("bert-sentiment-analysis")
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"]
                )
                if not runs.empty:
                    latest_run_id = runs.iloc[0]['run_id']
                    return f"runs:/{latest_run_id}/model"

            return None

        except Exception as e:
            logger.error(f"Error finding latest model: {e}")
            return None

    def load_fallback_model(self) -> bool:
        """Load fallback model from HuggingFace"""
        try:
            logger.info("Loading fallback model from HuggingFace...")

            model_name = "distilbert-base-uncased-finetuned-sst-2-english"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()

            self.model_info = {
                'version': 'fallback-distilbert-sst2',
                'model_uri': f'huggingface:{model_name}',
                'source': 'huggingface'
            }

            self.is_loaded = True
            logger.info("Fallback model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False

    def load_model(self, run_id: Optional[str] = None, model_version: Optional[str] = None) -> bool:
        """Main method to load model (try MLflow first, then fallback)"""
        # Try MLflow first
        if self.load_from_mlflow(run_id, model_version):
            return True

        # If MLflow fails, use fallback
        logger.warning("MLflow loading failed, using fallback model")
        return self.load_fallback_model()

    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on input text"""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded properly")

        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get prediction and confidence
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()

            sentiment = "positive" if predicted_class == 1 else "negative"
            latency_ms = (time.time() - start_time) * 1000

            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'latency_ms': latency_ms,
                'model_version': self.model_info.get('version', 'unknown')
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Prediction failed: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the model"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "is_loaded": self.is_loaded,
            "model_info": self.model_info
        }


class SentimentAPI:
    """Main API class that coordinates everything"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.monitor = ModelMonitor()
        self.app = FastAPI(title="BERT Sentiment Analysis API", version="1.0.0")
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.on_event("startup")
        async def startup_event():
            """Load model on startup"""
            logger.info("Starting BERT Sentiment Analysis API...")
            success = self.model_manager.load_model()
            if not success:
                logger.error("Failed to load any model. API will not function properly.")
            else:
                logger.info("Model loaded successfully. API ready!")

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_sentiment(request: PredictionRequest):
            """Predict sentiment for given text"""
            try:
                result = self.model_manager.predict(request.text)

                # Log prediction for monitoring
                self.monitor.log_prediction(
                    request.text,
                    result['sentiment'],
                    result['confidence'],
                    result['latency_ms'],
                    result['model_version']
                )

                response = PredictionResponse(
                    text=request.text,
                    sentiment=result['sentiment'],
                    confidence=result['confidence'],
                    prediction_time_ms=round(result['latency_ms'], 2),
                    model_version=result['model_version'],
                    timestamp=datetime.now().isoformat()
                )

                return response

            except ValueError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            health_status = self.model_manager.get_health_status()
            is_healthy = health_status["is_loaded"] and health_status["model_loaded"] and health_status[
                "tokenizer_loaded"]

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                **health_status,
                "timestamp": datetime.now().isoformat()
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Get monitoring metrics"""
            return self.monitor.get_stats()

        @self.app.post("/reload_model")
        async def reload_model(run_id: str = None):
            """Reload model from MLflow"""
            success = self.model_manager.load_model(run_id=run_id)
            if success:
                return {
                    "message": "Model reloaded successfully",
                    "model_info": self.model_manager.model_info
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to reload model")

        @self.app.get("/")
        async def root():
            """Root endpoint with API info"""
            health_status = self.model_manager.get_health_status()
            is_ready = health_status["is_loaded"] and health_status["model_loaded"] and health_status[
                "tokenizer_loaded"]

            return {
                "message": "BERT Sentiment Analysis API",
                "version": "1.0.0",
                "status": "ready" if is_ready else "loading",
                "endpoints": {
                    "predict": "/predict",
                    "health": "/health",
                    "metrics": "/metrics",
                    "reload": "/reload_model",
                    "docs": "/docs"
                },
                "model_info": self.model_manager.model_info
            }


# Initialize the API
sentiment_api = SentimentAPI()
app = sentiment_api.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)