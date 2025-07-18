# Makefile - Production MLOps with PostgreSQL Backend
.PHONY: help build setup run train test stop clean logs health info

# Default target
help:
	@echo "🚀 MLflow + PostgreSQL + BERT MLOps Pipeline"
	@echo "================================================"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup     - Build and start all services"
	@echo "  make train     - Train BERT model with PostgreSQL backend"
	@echo "  make test      - Test the API endpoints"
	@echo "  make monitor   - Run monitoring dashboard"
	@echo ""
	@echo "Development:"
	@echo "  make build     - Build Docker images"
	@echo "  make run       - Start all services"
	@echo "  make logs      - View all service logs"
	@echo "  make stop      - Stop all services"
	@echo "  make clean     - Clean up everything"
	@echo ""
	@echo "Database & Storage:"
	@echo "  make db-shell  - Connect to PostgreSQL"
	@echo "  make db-admin  - Open database admin interface"
	@echo "  make minio     - Open MinIO interface"
	@echo ""
	@echo "Monitoring:"
	@echo "  make health    - Check all service health"
	@echo "  make metrics   - Get current performance metrics"
	@echo "  make info      - Show all service URLs"

# Build Docker images
build:
	@echo "🔨 Building Docker images..."
	docker-compose -f docker-compose-postgres.yml build

# Setup complete environment
setup: build
	@echo "🚀 Setting up complete MLOps environment..."
	docker-compose -f docker-compose-postgres.yml up -d
	@echo "⏳ Waiting for services to be ready..."
	sleep 30
	docker-compose -f docker-compose-postgres.yml run --rm minio-setup
	@echo "✅ Environment ready!"
	@echo ""
	@echo "🌐 Access Points:"
	@echo "  📊 MLflow UI:      http://localhost:5000"
	@echo "  🚀 API Docs:       http://localhost:8000/docs"
	@echo "  🗄️ Database Admin: http://localhost:8080"
	@echo "  📦 MinIO Console:  http://localhost:9001"

# Start all services
run:
	@echo "🚀 Starting MLOps services with PostgreSQL..."
	docker-compose -f docker-compose-postgres.yml up -d
	@echo "⏳ Waiting for services..."
	sleep 15
	@echo "✅ Services started!"

# Train model with PostgreSQL backend
train:
	@echo "🤖 Training BERT model with PostgreSQL MLflow backend..."
	docker-compose -f docker-compose-postgres.yml exec bert-api python train_bert_postgres.py

# Test API endpoints
test:
	@echo "🧪 Testing API endpoints..."
	@echo "Testing health endpoint..."
	curl -s http://localhost:8000/health | python -m json.tool || echo "Health check failed"
	@echo ""
	@echo "Testing prediction endpoint..."
	curl -s -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"text": "I love this amazing product!"}' | python -m json.tool || echo "Prediction test failed"

# Comprehensive testing
test-full:
	@echo "🧪 Running comprehensive API tests..."
	docker-compose -f docker-compose-postgres.yml exec bert-api python test_api.py

# Monitor model performance
monitor:
	@echo "📊 Starting monitoring dashboard..."
	docker-compose -f docker-compose-postgres.yml exec bert-api python monitor_postgres.py

# Database operations
db-shell:
	@echo "🗄️ Connecting to PostgreSQL..."
	docker-compose -f docker-compose-postgres.yml exec postgres psql -U mlflow_user -d mlflow_db

db-admin:
	@echo "🌐 Starting database admin interface..."
	docker-compose -f docker-compose-postgres.yml --profile management up -d adminer
	@echo "📱 Adminer available at: http://localhost:8080"
	@echo "   Server: postgres"
	@echo "   Username: mlflow_user"
	@echo "   Password: mlflow_password"
	@echo "   Database: mlflow_db"

# MinIO interface
minio:
	@echo "📦 MinIO Console available at: http://localhost:9001"
	@echo "   Username: minio_user"
	@echo "   Password: minio_password"

# Health checks
health:
	@echo "🏥 Checking service health..."
	@echo "PostgreSQL:"
	@docker-compose -f docker-compose-postgres.yml exec postgres pg_isready -U mlflow_user -d mlflow_db && echo "✅ PostgreSQL healthy" || echo "❌ PostgreSQL unhealthy"
	@echo "MinIO:"
	@curl -s -f http://localhost:9000/minio/health/live > /dev/null && echo "✅ MinIO healthy" || echo "❌ MinIO unhealthy"
	@echo "MLflow:"
	@curl -s -f http://localhost:5000/health > /dev/null && echo "✅ MLflow healthy" || echo "❌ MLflow unhealthy"
	@echo "API:"
	@curl -s -f http://localhost:8000/health > /dev/null && echo "✅ API healthy" || echo "❌ API unhealthy"

# Get current metrics
metrics:
	@echo "📊 Current system metrics..."
	@echo "API Metrics:"
	@curl -s http://localhost:8000/metrics | python -m json.tool || echo "Could not get API metrics"

# Environment info
info:
	@echo "ℹ️  Environment Information"
	@echo "=========================="
	@echo "Services:"
	@echo "  📊 MLflow UI:      http://localhost:5000"
	@echo "  🚀 API Server:     http://localhost:8000"
	@echo "  🚀 API Docs:       http://localhost:8000/docs"
	@echo "  🗄️ Database Admin: http://localhost:8080 (run 'make db-admin' first)"
	@echo "  📦 MinIO Console:  http://localhost:9001"
	@echo ""
	@echo "Database Connection:"
	@echo "  Host: localhost:5432"
	@echo "  Database: mlflow_db"
	@echo "  Username: mlflow_user"
	@echo "  Password: mlflow_password"
	@echo ""
	@echo "Storage:"
	@echo "  Artifacts: s3://mlflow-artifacts/"
	@echo "  Endpoint: http://localhost:9000"

# View logs
logs:
	@echo "📋 Viewing all service logs..."
	docker-compose -f docker-compose-postgres.yml logs -f

logs-mlflow:
	@echo "📋 Viewing MLflow logs..."
	docker-compose -f docker-compose-postgres.yml logs -f mlflow

logs-api:
	@echo "📋 Viewing API logs..."
	docker-compose -f docker-compose-postgres.yml logs -f bert-api

logs-db:
	@echo "📋 Viewing PostgreSQL logs..."
	docker-compose -f docker-compose-postgres.yml logs -f postgres

# Stop services
stop:
	@echo "🛑 Stopping all services..."
	docker-compose -f docker-compose-postgres.yml down

# Clean up everything
clean: stop
	@echo "🧹 Cleaning up containers, volumes, and images..."
	docker-compose -f docker-compose-postgres.yml down -v
	docker system prune -f
	@echo "✅ Cleanup complete!"

# Backup database and artifacts
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "📊 Backing up PostgreSQL database..."
	docker-compose -f docker-compose-postgres.yml exec -T postgres pg_dump -U mlflow_user mlflow_db > backups/$(shell date +%Y%m%d_%H%M%S)/mlflow_db.sql
	@echo "✅ Backup created in backups/$(shell date +%Y%m%d_%H%M%S)/"

# Development helpers
dev-shell:
	@echo "🔧 Opening development shell..."
	docker-compose -f docker-compose-postgres.yml exec bert-api bash

# Quick metrics
status:
	@echo "📊 Quick Status Check..."
	@echo "Container Status:"
	@docker-compose -f docker-compose-postgres.yml ps
	@echo ""
	@echo "Service Health:"
	@make health

# Reset everything (careful!)
reset: clean
	@echo "🔄 Resetting everything..."
	@echo "⚠️  This will delete ALL data!"
	@read -p "Are you sure? Type 'yes' to continue: " confirm; \
	if [ "$confirm" = "yes" ]; then \
		docker volume rm bert-mlops-production_postgres_data bert-mlops-production_minio_data 2>/dev/null || true; \
		echo "✅ Reset complete!"; \
	else \
		echo "❌ Reset cancelled."; \
	fi