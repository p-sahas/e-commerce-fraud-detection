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
        mlflow-ui test lint format \
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
	@echo "    make all           Full pipeline (setup → data → train → test)"
	@echo "    make setup         Install dependencies and create directories"
	@echo "    make data          Run PySpark data pipeline"
	@echo "    make train         Train LR + LightGBM, register best model"
	@echo "    make train-lgbm    Train LightGBM only"
	@echo "    make train-lr      Train Logistic Regression only"
	@echo ""
	@echo "  Streaming:"
	@echo "    make stream-demo   Local demo (no Kafka needed)"
	@echo "    make stream        Start all three Kafka services (threaded)"
	@echo "    make producer      Kafka producer only"
	@echo "    make inference     Kafka inference consumer only"
	@echo "    make analytics     Kafka analytics consumer only"
	@echo ""
	@echo "  MLflow:"
	@echo "    make mlflow-ui     Launch MLflow UI at http://localhost:$(MLFLOW_PORT)"
	@echo ""
	@echo "  Quality:"
	@echo "    make test          Run pytest suite"
	@echo "    make lint          Lint with ruff + black --check"
	@echo "    make format        Auto-format with black"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean         Remove all generated artifacts"
	@echo "    make clean-data    Remove processed data only"
	@echo "    make clean-models  Remove model artifacts only"
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
# MLFLOW UI
# =============================================================================
mlflow-ui:
	@echo ""
	@echo "  ── Launching MLflow UI at http://localhost:$(MLFLOW_PORT) ──────"
	mlflow ui --backend-store-uri $(MLFLOW_URI) --port $(MLFLOW_PORT)

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
