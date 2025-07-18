-- Initialize MLflow database in your existing postgres
CREATE DATABASE mlflow_db;

-- Create MLflow user with your existing password pattern
CREATE USER mlflow_user WITH PASSWORD 'pass';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO postgres;

-- Connect to new database and set permissions
\c mlflow_db;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO mlflow_user;
GRANT ALL ON SCHEMA public TO postgres;

-- Grant table and sequence permissions (for future tables)
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow_user;
