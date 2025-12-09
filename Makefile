.PHONY: help install setup dev test clean docker-up docker-down

# Variables
PYTHON := poetry run python
PYTEST := poetry run pytest
UVICORN := poetry run uvicorn
BLACK := poetry run black
RUFF := poetry run ruff
MYPY := poetry run mypy

help:
	@echo "RAG System - Make Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install all dependencies"
	@echo "  make setup            Initialize database and collections"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start development server with reload"
	@echo "  make format           Format code with Black"
	@echo "  make lint             Run linting checks"
	@echo "  make type-check       Run type checking with MyPy"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        Start all services with Docker Compose"
	@echo "  make docker-down      Stop all Docker services"
	@echo "  make docker-logs      View Docker logs"
	@echo "  make docker-build     Build Docker image"
	@echo ""
	@echo "Data & Maintenance:"
	@echo "  make ingest           Run sample document ingestion"
	@echo "  make evaluate         Run evaluation on golden dataset"
	@echo "  make cleanup          Run cleanup maintenance tasks"
	@echo "  make backup           Create system backup"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Clean up temporary files"
	@echo "  make shell            Open Python shell with context"

# Installation
install:
	@echo "Installing dependencies..."
	poetry install
	@echo "Done!"

setup:
	@echo "Setting up RAG system..."
	$(PYTHON) scripts/setup.py
	@echo "System initialized!"

# Development
dev:
	@echo "Starting development server..."
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

format:
	@echo "Formatting code..."
	$(BLACK) src/ tests/ scripts/
	@echo "Code formatted!"

lint:
	@echo "Running linters..."
	$(RUFF) src/ tests/ scripts/
	@echo "Linting complete!"

type-check:
	@echo "Type checking..."
	$(MYPY) src/
	@echo "Type checking complete!"

# Testing
test:
	@echo "Running all tests..."
	$(PYTEST) tests/ -v

test-unit:
	@echo "Running unit tests..."
	$(PYTEST) tests/ -v -m "not integration"

test-integration:
	@echo "Running integration tests..."
	$(PYTEST) tests/test_integration.py -v

test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Docker
docker-up:
	@echo "Starting Docker services..."
	docker-compose -f deployment/docker-compose.yml up -d
	@echo "Services started! Waiting for initialization..."
	@sleep 10
	@echo "Checking health..."
	@curl -f http://localhost:8000/health || echo "API not ready yet"

docker-down:
	@echo "Stopping Docker services..."
	docker-compose -f deployment/docker-compose.yml down

docker-logs:
	docker-compose -f deployment/docker-compose.yml logs -f

docker-build:
	@echo "Building Docker image..."
	docker build -t rag-system:latest -f deployment/Dockerfile .

# Data & Maintenance
ingest:
	@echo "Ingesting sample documents..."
	$(PYTHON) scripts/ingest.py --dir data/sample_docs/

evaluate:
	@echo "Running evaluation..."
	$(PYTHON) scripts/evaluate.py --golden-dataset data/golden_dataset.jsonl

cleanup:
	@echo "Running cleanup tasks..."
	$(PYTHON) scripts/maintenance.py --cleanup

backup:
	@echo "Creating backup..."
	$(PYTHON) scripts/maintenance.py --backup

# Utilities
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Cleanup complete!"

shell:
	@echo "Opening Python shell..."
	$(PYTHON) -i -c "from src.core.config import settings; from src.core.embeddings import get_embedding_model; from src.core.vector_store import get_vector_store; print('RAG System Shell - Components loaded')"

# CI/CD helpers
ci-lint: format lint type-check

ci-test: test-cov

ci-all: ci-lint ci-test

# Quick start
quickstart: install docker-up setup
	@echo ""
	@echo "🚀 RAG System is ready!"
	@echo ""
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo ""
	@echo "Try: make dev"