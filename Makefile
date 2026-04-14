# =============================================================================
# E-Commerce Fraud Detection — Makefile
# =============================================================================
# One-command execution: make all
#
# Requirements (Windows):
#   Install 'make' via: choco install make  OR  scoop install make
#   Run from Git Bash, MSYS2, or any POSIX-compatible shell.
#
# Usage:
#   make help          — show this message
#   make all           — full pipeline: setup → data → train → test
#   make setup         — create directories and install Python deps
#   make data          — run PySpark data pipeline
#   make train         — train models and register best in MLflow
#   make train-lgbm    — train LightGBM only
#   make train-lr      — train Logistic Regression only
#   make stream-demo   — local streaming demo (no Kafka required)
#   make stream        — start all Kafka streaming services
#   make producer      — start Kafka producer only
#   make inference     — start Kafka inference consumer only
#   make analytics     — start Kafka analytics consumer only
#   make mlflow-ui     — open MLflow tracking UI at http://localhost:5000
#   make test          — run pytest suite
#   make lint          — run ruff + black check
#   make format        — auto-format with black
#   make clean         — remove all generated artifacts
#   make clean-data    — remove only processed data artifacts
#   make clean-models  — remove only model artifacts
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON        := python
PIP           := pip
PYTEST        := pytest
RUFF          := ruff
BLACK         := black
MLFLOW_COMPOSE := docker-compose.mlflow.yml

RAW_DATA      := data/raw/Fraud_Data.csv
PROCESSED_SENTINEL := artifacts/data/X_train.csv
MODEL_SENTINEL     := artifacts/models/best_model_meta.json

PIPELINES_DIR := pipelines
SRC_DIR       := src
TESTS_DIR     := tests
ANALYTICS_DIR := analytics

MLFLOW_PORT   := 5000
MLFLOW_URI    := file:./mlruns

# Phony targets — never treated as file names
.PHONY: all setup data train train-lgbm train-lr \
        stream stream-demo producer inference analytics \
        mlflow-start mlflow-stop mlflow-restart mlflow-status \
        mlflow-logs mlflow-ui mlflow-promote mlflow-compare mlflow-clean \
        test lint format \
        clean clean-data clean-models \
        dirs check-data help

# ── Default target ─────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║       E-Commerce Fraud Detection — Make Targets          ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make all              Full pipeline (setup → data → train → test)"
	@echo "    make setup            Install dependencies and create directories"
	@echo "    make data             Run PySpark data pipeline"
	@echo "    make train            Train LR + LightGBM, register best model"
	@echo "    make train-lgbm       Train LightGBM only"
	@echo "    make train-lr         Train Logistic Regression only"
	@echo ""
	@echo "  Streaming:"
	@echo "    make stream-demo      Local demo (no Kafka needed)"
	@echo "    make stream           Start all three Kafka services (threaded)"
	@echo "    make producer         Kafka producer only"
	@echo "    make inference        Kafka inference consumer only"
	@echo "    make analytics        Kafka analytics consumer only"
	@echo ""
	@echo "  MLflow Server:"
	@echo "    make mlflow-start     Start tracking server (Docker)"
	@echo "    make mlflow-stop      Stop tracking server"
	@echo "    make mlflow-restart   Restart tracking server"
	@echo "    make mlflow-status    Show server container status"
	@echo "    make mlflow-logs      Tail server logs"
	@echo "    make mlflow-ui        Open UI at http://localhost:$(MLFLOW_PORT)"
	@echo "    make mlflow-promote   Promote Staging model → Production"
	@echo "    make mlflow-compare   Print run comparison table"
	@echo "    make mlflow-clean     Delete local mlruns/ directory"
	@echo ""
	@echo "  Quality:"
	@echo "    make test             Run pytest suite with coverage"
	@echo "    make test-fast        Run tests without coverage"
	@echo "    make lint             Lint with ruff + black --check"
	@echo "    make format           Auto-format with black"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean            Remove all generated artifacts"
	@echo "    make clean-data       Remove processed data only"
	@echo "    make clean-models     Remove model artifacts only"
	@echo ""

# =============================================================================
# SETUP
# =============================================================================
setup: dirs
	@echo ""
	@echo "  ── Installing Python dependencies ──────────────────────────"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "  ✓ Setup complete"

dirs:
	@echo "  ── Creating required directories ────────────────────────────"
	$(PYTHON) -c "\
import os; \
dirs = ['data/raw', 'data/processed', 'artifacts/data', 'artifacts/models', \
        'artifacts/plots', 'artifacts/predictions', \
        'analytics', 'mlruns', 'tests']; \
[os.makedirs(d, exist_ok=True) for d in dirs]; \
print('  ✓ Directories ready')"

# =============================================================================
# DATA PIPELINE
# =============================================================================
check-data:
	@$(PYTHON) -c "\
import os, sys; \
path = '$(RAW_DATA)'; \
sys.exit(0) if os.path.exists(path) \
else (print(f'  ERROR: Raw data not found at {path}'), sys.exit(1))"

data: dirs check-data
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║            RUNNING DATA PIPELINE                         ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	$(PYTHON) $(PIPELINES_DIR)/data_pipeline.py
	@echo "  ✓ Data pipeline complete → artifacts/data/"

# Re-run even if output already exists
data-force: dirs check-data
	@echo ""
	@echo "  ── Force rebuilding processed data ─────────────────────────"
	$(PYTHON) -c "\
import sys; sys.path.insert(0,'$(PIPELINES_DIR)'); \
sys.path.insert(0,'utils'); \
from data_pipeline import data_pipeline; \
data_pipeline(force_rebuild=True)"
	@echo "  ✓ Data pipeline (forced) complete"

# =============================================================================
# TRAINING PIPELINE
# =============================================================================
train: $(PROCESSED_SENTINEL)
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║            RUNNING TRAINING PIPELINE                     ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	$(PYTHON) $(PIPELINES_DIR)/training_pipeline.py \
		--models logistic_regression lightgbm \
		--min-auc 0.75
	@echo "  ✓ Training complete → artifacts/models/best_model_meta.json"

train-lgbm: $(PROCESSED_SENTINEL)
	@echo "  ── Training LightGBM only ───────────────────────────────────"
	$(PYTHON) $(PIPELINES_DIR)/training_pipeline.py \
		--models lightgbm \
		--min-auc 0.75
	@echo "  ✓ LightGBM training complete"

train-lr: $(PROCESSED_SENTINEL)
	@echo "  ── Training Logistic Regression only ────────────────────────"
	$(PYTHON) $(PIPELINES_DIR)/training_pipeline.py \
		--models logistic_regression \
		--min-auc 0.75
	@echo "  ✓ Logistic Regression training complete"

train-force: $(PROCESSED_SENTINEL)
	@echo "  ── Force retraining all models ──────────────────────────────"
	$(PYTHON) $(PIPELINES_DIR)/training_pipeline.py \
		--models logistic_regression lightgbm \
		--min-auc 0.75 \
		--force-retrain
	@echo "  ✓ Force retrain complete"

# Sentinel: only run data pipeline if processed files are missing
$(PROCESSED_SENTINEL):
	@echo "  Processed data not found — running data pipeline first …"
	$(MAKE) data

# =============================================================================
# STREAMING PIPELINE
# =============================================================================
stream-demo:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║        STREAMING DEMO  (no Kafka required)               ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	$(PYTHON) $(PIPELINES_DIR)/streaming_inference_pipeline.py demo 500
	@echo "  ✓ Demo complete → analytics/"

stream:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║   STARTING ALL STREAMING SERVICES  (Ctrl+C to stop)     ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo "  Ensure Kafka is running: docker-compose up -d zookeeper kafka"
	$(PYTHON) $(PIPELINES_DIR)/streaming_inference_pipeline.py all

producer:
	@echo "  ── Starting Kafka producer ──────────────────────────────────"
	$(PYTHON) $(PIPELINES_DIR)/streaming_inference_pipeline.py producer

inference:
	@echo "  ── Starting inference consumer ──────────────────────────────"
	$(PYTHON) $(PIPELINES_DIR)/streaming_inference_pipeline.py inference

analytics:
	@echo "  ── Starting analytics consumer ──────────────────────────────"
	$(PYTHON) $(PIPELINES_DIR)/streaming_inference_pipeline.py analytics

# =============================================================================
# MLFLOW SERVER (Docker-based tracking server)
# =============================================================================

## Start the MLflow Postgres-backed tracking server in the background
mlflow-start:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║          STARTING MLFLOW TRACKING SERVER                 ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	docker-compose -f $(MLFLOW_COMPOSE) up -d --remove-orphans
	@echo ""
	@echo "  Waiting for server to be healthy …"
	@$(PYTHON) -c "\
import urllib.request, time, sys; \
for i in range(30): \
    try: \
        urllib.request.urlopen('http://localhost:5001/health', timeout=2); \
        print('  ✓ MLflow UI ready at http://localhost:5001'); \
        sys.exit(0) \
    except Exception: \
        time.sleep(2) \
print('  ✗ Server did not respond in 60s — check: make mlflow-logs'); \
sys.exit(1)"

## Stop the MLflow tracking server
mlflow-stop:
	@echo "  ── Stopping MLflow tracking server ──────────────────────────"
	docker-compose -f $(MLFLOW_COMPOSE) down
	@echo "  ✓ MLflow server stopped"

## Restart the server (stops then starts)
mlflow-restart: mlflow-stop mlflow-start

## Show container status
mlflow-status:
	@echo "  ── MLflow container status ──────────────────────────────────"
	docker-compose -f $(MLFLOW_COMPOSE) ps

## Tail server logs
mlflow-logs:
	docker-compose -f $(MLFLOW_COMPOSE) logs -f mlflow

## Open the MLflow UI in the default browser
## Falls back to just printing the URL if no browser is detected
mlflow-ui:
	@echo ""
	@echo "  ── MLflow UI → http://localhost:$(MLFLOW_PORT) ─────────────────"
	@$(PYTHON) -c "\
import webbrowser, sys; \
opened = webbrowser.open('http://localhost:$(MLFLOW_PORT)', new=2); \
print('  Browser opened.' if opened else '  Open manually: http://localhost:$(MLFLOW_PORT)')"

## Promote the latest Staging model to Production
## Override version with: make mlflow-promote MLFLOW_VERSION=3
MLFLOW_VERSION ?=
mlflow-promote:
	@echo ""
	@echo "  ── Promoting model to Production ────────────────────────────"
	$(PYTHON) src/mlflow_utils.py promote --stage Production \
		$(if $(MLFLOW_VERSION),--version $(MLFLOW_VERSION),)
	@echo "  ✓ Promotion complete"

## Print a run comparison table for the experiment
mlflow-compare:
	@echo ""
	@echo "  ── Comparing MLflow runs ────────────────────────────────────"
	$(PYTHON) src/mlflow_utils.py compare

## Delete the local mlruns/ directory (only for file-based tracking)
mlflow-clean:
	@echo "  ── Removing local mlruns/ ───────────────────────────────────"
	$(PYTHON) -c "\
import shutil; \
shutil.rmtree('mlruns', ignore_errors=True); \
print('  ✓ mlruns/ removed')"

# =============================================================================
# TESTING & QUALITY
# =============================================================================
test:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║                RUNNING TEST SUITE                        ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	$(PYTEST) $(TESTS_DIR)/ \
		-v \
		--tb=short \
		--cov=$(SRC_DIR) \
		--cov=pipelines \
		--cov-report=term-missing \
		--cov-report=html:artifacts/coverage_html
	@echo "  ✓ Tests complete — coverage report → artifacts/coverage_html/"

test-fast:
	@echo "  ── Running tests (no coverage) ─────────────────────────────"
	$(PYTEST) $(TESTS_DIR)/ -v --tb=short -x

lint:
	@echo ""
	@echo "  ── Linting with ruff ────────────────────────────────────────"
	$(RUFF) check $(SRC_DIR)/ $(PIPELINES_DIR)/ utils/
	@echo "  ── Checking formatting with black ───────────────────────────"
	$(BLACK) --check $(SRC_DIR)/ $(PIPELINES_DIR)/ utils/
	@echo "  ✓ Lint passed"

format:
	@echo "  ── Auto-formatting with black ───────────────────────────────"
	$(BLACK) $(SRC_DIR)/ $(PIPELINES_DIR)/ utils/
	@echo "  ✓ Formatting complete"

# =============================================================================
# FULL PIPELINE (make all)
# =============================================================================
all: setup data train test
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║              FULL PIPELINE COMPLETE ✓                    ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Artifacts:"
	@echo "    Processed data  → artifacts/data/"
	@echo "    Trained models  → artifacts/models/"
	@echo "    Evaluation plots→ artifacts/plots/"
	@echo "    MLflow runs     → mlruns/"
	@echo ""
	@echo "  Next steps:"
	@echo "    make mlflow-ui      View experiment results"
	@echo "    make stream-demo    Test streaming inference"
	@echo ""

# =============================================================================
# CLEANUP
# =============================================================================
clean: clean-data clean-models
	@echo "  ── Removing additional artifacts ────────────────────────────"
	$(PYTHON) -c "\
import shutil, os; \
targets = ['artifacts/plots', 'artifacts/predictions', \
           'artifacts/coverage_html', 'analytics', \
           '.pytest_cache', '__pycache__']; \
[shutil.rmtree(t, ignore_errors=True) for t in targets]; \
[os.remove(f) for f in ['pipeline.log'] if os.path.exists(f)]; \
print('  ✓ Clean complete')"

clean-data:
	@echo "  ── Removing processed data ──────────────────────────────────"
	$(PYTHON) -c "\
import shutil; \
shutil.rmtree('artifacts/data', ignore_errors=True); \
shutil.rmtree('data/processed', ignore_errors=True); \
print('  ✓ Processed data removed')"

clean-models:
	@echo "  ── Removing model artifacts ─────────────────────────────────"
	$(PYTHON) -c "\
import shutil; \
shutil.rmtree('artifacts/models', ignore_errors=True); \
shutil.rmtree('mlruns', ignore_errors=True); \
print('  ✓ Model artifacts removed')"
