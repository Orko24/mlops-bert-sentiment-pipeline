# train_bert_postgres.py - Class-based architecture
import mlflow
import mlflow.pytorch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import time
import os
from datetime import datetime
import random
from typing import Optional, Dict, Any, List, Tuple


class MLflowManager:
    """Handles MLflow configuration and connection"""

    def __init__(self, mlflow_uri: str = "http://localhost:5000",
                 experiment_name: str = "bert-sentiment-analysis"):
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.is_connected = False
        self.current_run = None

    def setup_connection(self) -> bool:
        """Setup MLflow connection and experiment"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_uri)

            # Configure artifact storage (MinIO/S3)
            self._configure_artifact_storage()

            print(f"🔗 MLflow Tracking URI: {self.mlflow_uri}")

            # Test connection
            mlflow.get_tracking_uri()
            print("✅ MLflow connection successful")

            # Setup experiment
            self.experiment_id = self._setup_experiment()
            self.is_connected = True
            return True

        except Exception as e:
            print(f"⚠️  MLflow connection warning: {e}")
            print("🔧 Continuing with local fallback...")
            self.is_connected = False
            return False

    def _configure_artifact_storage(self):
        """Configure S3/MinIO artifact storage"""
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio_user")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_password")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

    def _setup_experiment(self) -> Optional[str]:
        """Setup or create experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"📝 Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"📂 Using existing experiment: {self.experiment_name}")

            mlflow.set_experiment(self.experiment_name)
            return experiment_id

        except Exception as e:
            print(f"⚠️  Experiment setup failed: {e}")
            print("📝 Using default experiment")
            return "0"

    def start_run(self) -> Optional[str]:
        """Start MLflow run"""
        if not self.is_connected:
            return None

        try:
            self.current_run = mlflow.start_run()
            run_id = self.current_run.info.run_id
            print(f"🚀 Starting MLflow run: {run_id}")
            return run_id
        except Exception as e:
            print(f"⚠️  MLflow run start failed: {e}")
            return None

    def end_run(self):
        """End MLflow run"""
        if self.current_run:
            try:
                mlflow.end_run()
                self.current_run = None
            except Exception:
                pass

    def log_params(self, params: Dict[str, Any]) -> bool:
        """Log parameters to MLflow"""
        if not self.is_connected:
            return False

        try:
            mlflow.log_params(params)
            return True
        except Exception as e:
            print(f"⚠️  Parameter logging failed: {e}")
            return False

    def log_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Log metrics to MLflow"""
        if not self.is_connected:
            return False

        try:
            mlflow.log_metrics(metrics)
            return True
        except Exception as e:
            print(f"⚠️  Metrics logging failed: {e}")
            return False

    def log_model(self, model, artifact_path: str = "model") -> bool:
        """Log model to MLflow"""
        if not self.is_connected:
            return False

        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path
            )
            return True
        except Exception as e:
            print(f"⚠️  Model logging failed: {e}")
            return False

    def register_model(self, run_id: str, model_name: str = "bert-sentiment-classifier") -> bool:
        """Register model in MLflow registry"""
        if not self.is_connected:
            return False

        try:
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=model_name
            )
            print(f"📝 Model registered: {model_name} v{model_version.version}")
            return True
        except Exception as e:
            print(f"⚠️  Model registration failed: {e}")
            return False


class DataManager:
    """Handles data preparation and generation"""

    @staticmethod
    def generate_sample_data(num_samples: int = 200) -> Tuple[List[str], List[int]]:
        """Generate sample sentiment data"""
        positive_texts = [
                             "I love this product, it's amazing!",
                             "Great quality and fast delivery!",
                             "Excellent customer service!",
                             "Highly recommend this to everyone!",
                             "Best purchase I've made this year!",
                             "Outstanding performance and value!",
                             "Perfect solution for my needs!",
                             "Incredible results, very satisfied!"
                         ] * (num_samples // 16 + 1)

        negative_texts = [
                             "This product is terrible and broken",
                             "Worst customer service ever experienced",
                             "Complete waste of money, very disappointed",
                             "Poor quality, broke immediately",
                             "Would not recommend to anyone",
                             "Overpriced and underdelivered",
                             "Regret buying this completely",
                             "Terrible experience, avoid this!"
                         ] * (num_samples // 16 + 1)

        # Combine and shuffle
        texts = positive_texts[:num_samples // 2] + negative_texts[:num_samples // 2]
        labels = [1] * (num_samples // 2) + [0] * (num_samples // 2)

        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)

        return list(texts), list(labels)

    @staticmethod
    def split_data(texts: List[str], labels: List[int],
                   train_ratio: float = 0.8) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Split data into train and validation sets"""
        split_idx = int(train_ratio * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:]

        return train_texts, val_texts, train_labels, val_labels


class ModelManager:
    """Handles model initialization, training, and evaluation"""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def initialize_model(self):
        """Initialize model and tokenizer"""
        print("🤖 Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )

    def prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Prepare data for BERT training"""
        # Create dataset
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': labels
        })

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=128
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def save_model(self, model_path: str = "bert_model"):
        """Save model and tokenizer locally"""
        print("💾 Saving model...")
        os.makedirs(model_path, exist_ok=True)

        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)


class TrainingConfig:
    """Training configuration class"""

    def __init__(self,
                 epochs: int = 2,
                 batch_size: int = 8,
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 100,
                 weight_decay: float = 0.01):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay
        }


class BERTTrainer:
    """Main training orchestrator"""

    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 mlflow_uri: str = "http://localhost:5000",
                 experiment_name: str = "bert-sentiment-analysis"):

        # Initialize components
        self.mlflow_manager = MLflowManager(mlflow_uri, experiment_name)
        self.model_manager = ModelManager(model_name)
        self.data_manager = DataManager()

        # Setup MLflow connection
        self.mlflow_manager.setup_connection()

    def train(self,
              train_texts: List[str],
              train_labels: List[int],
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None,
              config: Optional[TrainingConfig] = None) -> Optional[str]:
        """Main training method"""

        if config is None:
            config = TrainingConfig()

        # Start MLflow run
        run_id = self.mlflow_manager.start_run()

        try:
            # Log parameters
            params = config.to_dict()
            params.update({
                "model_name": self.model_manager.model_name,
                "train_size": len(train_texts),
                "val_size": len(val_texts) if val_texts else 0,
                "timestamp": datetime.now().isoformat()
            })
            self.mlflow_manager.log_params(params)

            # Initialize model
            self.model_manager.initialize_model()

            # Prepare datasets
            print("📊 Preparing datasets...")
            train_dataset = self.model_manager.prepare_dataset(train_texts, train_labels)
            val_dataset = None
            if val_texts and val_labels:
                val_dataset = self.model_manager.prepare_dataset(val_texts, val_labels)

            # Setup training
            trainer = self._setup_trainer(config, train_dataset, val_dataset)

            # Train model
            print("🚀 Starting training...")
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time

            print(f"✅ Training completed in {training_time / 60:.2f} minutes")

            # Log training metrics
            self.mlflow_manager.log_metrics({
                "train_loss": train_result.training_loss,
                "training_time_minutes": training_time / 60,
            })

            # Evaluate model
            if val_dataset:
                eval_metrics = self._evaluate_model(trainer)
                self.mlflow_manager.log_metrics(eval_metrics)

            # Save model locally
            self.model_manager.save_model()

            # Log model to MLflow
            self.mlflow_manager.log_model(self.model_manager.model)

            # Register model
            if run_id:
                self.mlflow_manager.register_model(run_id)

            print("✅ Training pipeline completed successfully!")
            return run_id

        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        finally:
            self.mlflow_manager.end_run()

    def _setup_trainer(self, config: TrainingConfig,
                       train_dataset: Dataset,
                       val_dataset: Optional[Dataset]) -> Trainer:
        """Setup Hugging Face trainer"""

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="accuracy" if val_dataset else None,
            report_to=None  # Disable other tracking
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.model_manager.tokenizer)

        trainer = Trainer(
            model=self.model_manager.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.model_manager.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.model_manager.compute_metrics if val_dataset else None,
        )

        return trainer

    def _evaluate_model(self, trainer: Trainer) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        print("📈 Evaluating model...")
        eval_result = trainer.evaluate()
        print(f"📊 Validation Results: {eval_result}")

        # Format metrics for MLflow
        formatted_metrics = {}
        for key, value in eval_result.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', 'val_')
                formatted_metrics[metric_name] = value

        return formatted_metrics

    def train_with_sample_data(self,
                               num_samples: int = 200,
                               config: Optional[TrainingConfig] = None) -> Optional[str]:
        """Train with generated sample data"""
        print("📊 Generating sample data...")
        texts, labels = self.data_manager.generate_sample_data(num_samples)

        # Split data
        train_texts, val_texts, train_labels, val_labels = self.data_manager.split_data(
            texts, labels, train_ratio=0.8
        )

        print(f"📈 Train samples: {len(train_texts)}")
        print(f"📊 Validation samples: {len(val_texts)}")

        return self.train(train_texts, train_labels, val_texts, val_labels, config)


class TrainingPipeline:
    """High-level training pipeline interface"""

    @staticmethod
    def run_quick_training() -> Optional[str]:
        """Run a quick training for demo purposes"""
        print("🚀 BERT Training with MLflow Integration")
        print("=" * 50)

        # Initialize trainer
        trainer = BERTTrainer()

        # Configure for quick training
        config = TrainingConfig(
            epochs=1,  # Quick training for demo
            batch_size=8,
            learning_rate=2e-5
        )

        try:
            run_id = trainer.train_with_sample_data(
                num_samples=200,
                config=config
            )

            print(f"\n🎉 Training completed successfully!")
            print(f"🆔 Run ID: {run_id}")
            print(f"🌐 MLflow UI: http://localhost:5000")

            return run_id

        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("🔧 Check your MLflow server connection and try again")
            return None

    @staticmethod
    def run_full_training(epochs: int = 3, batch_size: int = 16) -> Optional[str]:
        """Run full training with custom parameters"""
        print("🚀 BERT Full Training with MLflow Integration")
        print("=" * 50)

        trainer = BERTTrainer()

        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=2e-5
        )

        try:
            run_id = trainer.train_with_sample_data(
                num_samples=1000,  # More data for full training
                config=config
            )

            print(f"\n🎉 Full training completed successfully!")
            print(f"🆔 Run ID: {run_id}")
            print(f"🌐 MLflow UI: http://localhost:5000")

            return run_id

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None


def main():
    """Main function"""
    # Run quick training by default
    TrainingPipeline.run_quick_training()


if __name__ == "__main__":
    main()