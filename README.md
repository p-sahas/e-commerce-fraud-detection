# E-Commerce Fraud Detection System

### End-to-End Machine Learning Pipeline with Real-Time Streaming

[![CI](https://github.com/p-sahas/e-commerce-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/p-sahas/e-commerce-fraud-detection/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PySpark 3.5](https://img.shields.io/badge/pyspark-3.5.1-orange.svg)](https://spark.apache.org/)
[![MLflow](https://img.shields.io/badge/mlflow-2.12+-green.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

A production-grade fraud detection system for e-commerce transactions, implementing the full ML lifecycle from raw data ingestion through real-time streaming inference.

**Key capabilities:**

- **PySpark preprocessing pipeline** — schema validation, feature engineering, leakage-safe scaling
- **Two-model training** — Logistic Regression (baseline) vs. LightGBM (advanced) with TimeSeriesSplit cross-validation
- **Cost-optimised thresholding** — minimises FP × $5 + FN × $100 business cost
- **MLflow experiment tracking** — full parameter / metric / artifact logging with Model Registry
- **Airflow orchestration** — 3 production DAGs with dependency sensing and AUC gate
- **Kafka streaming inference** — real-time fraud scoring with rolling velocity features (1h / 24h windows)
- **CI/CD** — GitHub Actions with lint, unit, Spark, and smoke-test gates

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        E-Commerce Fraud Detection                           │
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐    │
│  │  Raw CSV    │───>│  PySpark         │───>│  MLflow Model Registry   │    │
│  │  Fraud_Data │    │  Data Pipeline   │    │  (LR / LightGBM)         │    │
│  └─────────────┘    │  (Airflow DAG)   │    └────────────┬─────────────┘    │
│                     └──────────────────┘                 │                  │
│                                                          ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Kafka Streaming Pipeline                         │    │
│  │                                                                     │    │
│  │  CSV ──> Producer ──> [purchases] ──> Inference Consumer            │    │
│  │                                            │  (rolling features)    │    │
│  │                                            ▼                        │    │
│  │                                      [fraud_scores] ──> Analytics   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Infrastructure: Airflow :8081 · MLflow :5001 · Kafka UI :8080              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
e-commerce-fraud-detection/
│
├── data/raw/                      # Raw Kaggle dataset (Fraud_Data.csv)
├── config/
│   ├── config.yaml                # All pipeline settings (models, streaming, paths)
│   └── schema.json                # Transaction schema for validation
│
├── src/                           # Core library modules
│   ├── preprocessing.py           # PySpark ETL: clean, impute, engineer, scale
│   ├── feature_binning.py         # BucketizerBinningStrategy (age, purchase_value)
│   ├── outlier_detection.py       # IQR / Z-score capping
│   ├── training.py                # LR + LightGBM, threshold optimiser, AUC gate
│   └── mlflow_utils.py            # Experiment setup, run context manager, registry
│
├── pipelines/
│   ├── data_pipeline.py           # PySpark orchestration entry point
│   ├── training_pipeline.py       # End-to-end model training script
│   ├── batch_inference_pipeline.py# Batch scoring against MLflow registry
│   └── streaming_inference_pipeline.py  # Kafka producer + inference + analytics
│
├── dags/                          # Apache Airflow DAGs
│   ├── data_pipeline_dag.py       # Daily: ingest -> validate -> curate
│   ├── training_dag.py            # Weekly: train -> AUC gate -> register
│   └── batch_inference_dag.py     # Daily: load model -> score -> export
│
├── tests/
│   ├── conftest.py                # Shared fixtures (Spark session, event data)
│   ├── test_preprocessing.py      # PySpark preprocessing unit tests
│   ├── test_training.py           # AUC gate, threshold optimizer, model smoke tests
│   ├── test_streaming.py          # RollingFeatureStore, FraudModelLoader, producer
│   └── test_mlflow_utils.py       # Experiment setup, run context, log helpers
│
├── utils/
│   ├── config.py                  # Config loader (config.yaml -> Python dict)
│   └── spark_utils.py             # SparkSession factory, DataFrame helpers
│
├── docker-compose.yml             # Full stack: Kafka + Airflow + MLflow + Postgres
├── Dockerfile                     # Streaming service image (Python 3.11-slim)
├── Makefile                       # One-command pipeline execution
├── requirements.txt               # Python dependencies
├── ruff.toml                      # Linting configuration
├── pytest.ini                     # Test discovery and markers
├── .env.example                   # Environment variable template (copy -> .env)
└── .github/workflows/
    ├── ci.yml                     # Lint -> Unit -> Spark -> Smoke -> Quality gate
    ├── cd.yml                     # Docker build + push to ghcr.io on main merge
    └── security.yml               # pip-audit + bandit + TruffleHog (weekly)
```

---

## Quick Start

### Prerequisites

| Tool           | Version          | Install                                                            |
| -------------- | ---------------- | ------------------------------------------------------------------ |
| Python         | 3.11+            | [python.org](https://www.python.org/)                              |
| Java           | 11 (for PySpark) | `brew install openjdk@11` / [adoptium.net](https://adoptium.net/)  |
| Docker Desktop | 4.x+             | [docker.com](https://www.docker.com/products/docker-desktop/)      |
| GNU Make       | any              | Windows:[chocolatey](https://chocolatey.org/) `choco install make` |

### 1 — Clone & configure environment

```bash
git clone https://github.com/p-sahas/e-commerce-fraud-detection.git
cd e-commerce-fraud-detection

# Copy the environment template and fill in your values
cp .env.example .env
# Edit .env — at minimum set MLFLOW_TRACKING_URI and AWS credentials if using S3
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Run the full local pipeline (no Docker needed)

```bash
make all
# Runs: setup -> data pipeline -> training -> tests
```

Or run each step individually:

```bash
make data          # PySpark preprocessing -> artifacts/data/
make train         # Train LR + LightGBM, register best model
make stream-demo   # Streaming demo (100 events, no Kafka required)
make test          # Full pytest suite with coverage report
```

---

## Docker Stack

Start the full infrastructure in one command:

```bash
# Minimal setup: Kafka + MLflow only (recommended for development)
make infra-start

# Full stack: + Apache Airflow
make stack-start

# Check everything is running
make stack-status
```

### Service URLs

| Service          | URL                                            | Credentials       |
| ---------------- | ---------------------------------------------- | ----------------- |
| **Airflow UI**   | [http://localhost:8081](http://localhost:8081) | `admin` / `admin` |
| **MLflow UI**    | [http://localhost:5001](http://localhost:5001) | —                 |
| **Kafka UI**     | [http://localhost:8080](http://localhost:8080) | —                 |
| **Kafka broker** | `localhost:9093`                               | — (host access)   |

> **First-time Airflow setup:** Run `make airflow-init` once before `make airflow-start` to initialise the database and create the admin user.

### Recommended startup sequence

```bash
make infra-start      # Start Kafka + MLflow (wait ~30s)
make train            # Train model and register to MLflow
make stream-docker    # Launch streaming services (producer + inference + analytics)
make stack-start      # Add Airflow to orchestrate all DAGs
```

---

## Streaming Pipeline

The Kafka streaming pipeline consists of three independent services:

| Service                | Topic                          | Role                                                |
| ---------------------- | ------------------------------ | --------------------------------------------------- |
| **Producer**           | ->`purchases`                  | Replays CSV sorted by `purchase_time`               |
| **Inference Consumer** | `purchases` -> `fraud_scores`  | Computes rolling features, scores with MLflow model |
| **Analytics Consumer** | `fraud_scores` -> SQLite / CSV | Aggregates hourly fraud rates, triggers alerts      |

**Run locally (no Kafka):**

```bash
make stream-demo          # 500-event in-process demo
```

**Run with Docker Kafka:**

```bash
make kafka-start          # Start Zookeeper + Kafka + Kafka UI
make stream-docker        # Launch producer + inference + analytics containers
make kafka-topics         # List topics (purchases, fraud_scores, fraud_alerts)
make kafka-logs           # Tail broker logs
```

---

## MLflow Experiment Tracking

All training runs are logged to MLflow with:

- **Parameters** — model hyperparameters, CV splits, threshold range
- **Metrics** — AUC-ROC, AUC-PR, Precision, Recall, F1, expected cost, cost savings
- **Artifacts** — ROC curve, PR curve, confusion matrix, threshold cost curve, model files
- **Model signatures** — input/output schemas for production validation

```bash
# Start the MLflow Tracking server
make mlflow-start

# Open the MLflow UI
make mlflow-ui            # Opens http://localhost:5001

# Promote a model from Staging -> Production
make mlflow-promote

# Compare all runs in a table
make mlflow-compare
```

### Model Registry stages

```
None -> Staging -> Production -> Archived
```

The training DAG enforces an **AUC ≥ 0.75 quality gate** - models that fail are not registered.

---

## Airflow DAGs

Three DAGs orchestrate the complete pipeline lifecycle:

| DAG                   | Schedule             | Tasks                                                                                         |
| --------------------- | -------------------- | --------------------------------------------------------------------------------------------- |
| `data_pipeline_dag`   | Daily 02:00 UTC      | `wait_for_raw_data -> schema_validate -> quality_report -> curated_export`                    |
| `training_dag`        | Weekly Mon 03:00 UTC | `wait_for_data_dag -> load_curated -> feature_build -> kfold_train -> eval -> register_model` |
| `batch_inference_dag` | Daily 04:00 UTC      | `load_model -> score_batch -> export_scores`                                                  |

### Required Airflow Connections

Create these in **Admin -> Connections** before enabling the DAGs:

| Connection ID      | Type                | Notes                                      |
| ------------------ | ------------------- | ------------------------------------------ |
| `postgres_default` | Postgres            | Points to your RDS/local Postgres instance |
| `aws_default`      | Amazon Web Services | Used for S3 artifact upload (if enabled)   |

### DAG design principles

- `max_active_runs=1` - prevents concurrent run conflicts
- `ExternalTaskSensor` - training DAG waits for data DAG to complete
- XCom used for small values only; large artifacts persisted to `artifacts/`
- All tasks are **idempotent** - safe to retry on failure

---

## Testing

```bash
# Full suite with coverage HTML report
make test

# Fast run (no coverage overhead)
make test-fast

# Non-Spark tests only (fastest, ~5s)
pytest tests/test_streaming.py tests/test_mlflow_utils.py tests/test_training.py

# Spark tests only (requires Java 11, ~60s)
pytest tests/test_preprocessing.py
```

### Test coverage by module

| Test file               | Module tested                     | Key scenarios                                                                             |
| ----------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| `test_preprocessing.py` | `src/preprocessing.py`            | Feature engineering, imputation, split ratios, leakage-safe scaling                       |
| `test_training.py`      | `src/training.py`                 | AUC gate, threshold cost optimisation, evaluate_model, LR + LightGBM smoke tests          |
| `test_streaming.py`     | `streaming_inference_pipeline.py` | Rolling feature store (thread-safe), leakage exclusion, stub predictor, timestamp parsing |
| `test_mlflow_utils.py`  | `src/mlflow_utils.py`             | Experiment setup, run context manager (FAILED tag on exception), log_params batching      |

---

## CI/CD

### GitHub Actions workflows

| Workflow     | Trigger             | Jobs                                                                                |
| ------------ | ------------------- | ----------------------------------------------------------------------------------- |
| **CI**       | Push / PR           | lint -> unit-tests (60% coverage gate) -> spark-tests -> smoke-test -> quality-gate |
| **CD**       | Merge to `main`     | Build multi-arch Docker image -> push to `ghcr.io` -> create GitHub Release         |
| **Security** | Weekly +`main` push | pip-audit CVE scan -> bandit SAST -> TruffleHog secret scan                         |

### CI job graph

```
lint ──┬──► unit-tests ──► smoke-test ──┐
       │                                ├──► quality-gate (merge guard)
       └──► spark-tests ────────────────┘
```

All jobs must pass before a PR can be merged. The quality gate fails the build if any upstream job fails.

---

## Configuration

All pipeline behaviour is controlled via `config/config.yaml`. Key sections:

```yaml
training:
  cost_matrix:
    fp_cost: 5 # $ cost per false positive (manual review)
    fn_cost: 100 # $ cost per false negative (missed fraud)
  min_auc_gate: 0.75 # Models below this AUC are not registered

streaming:
  kafka_bootstrap_servers: "localhost:9092"
  window_1h_secs: 3600
  window_24h_secs: 86400
  alert_rate_threshold: 0.15 # flag hour if fraud rate > 15%

mlflow:
  experiment_name: "E-Commerce Fraud Detection"
  model_registry_name: "fraud_detection"
```

Override any value via environment variable - see `.env.example` for the full list.

---

## Makefile Reference

```bash
# Pipeline
make all              # Full pipeline: setup -> data -> train -> test
make data             # PySpark data pipeline
make train            # Train both models, register best
make train-lgbm       # LightGBM only
make train-lr         # Logistic Regression only

# Streaming
make stream-demo      # In-process demo (no Kafka)
make stream           # All 3 services in threads
make stream-docker    # Docker containers

# Infrastructure
make infra-start      # Kafka + MLflow
make stack-start      # Full stack (+ Airflow)
make stack-stop       # Stop everything
make kafka-topics     # List Kafka topics
make airflow-ui       # Open Airflow in browser

# MLflow
make mlflow-start / stop / ui / promote / compare

# Quality
make test             # pytest + coverage HTML
make lint             # ruff + black check
make format           # black auto-format

# Cleanup
make clean            # Remove all generated artifacts
make clean-data       # Processed data only
make clean-models     # Model artifacts only
```

---

## Dataset

**Source:** [Kaggle — E-Commerce Fraud Detection](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)

| Property   | Value                               |
| ---------- | ----------------------------------- |
| Records    | ~151,112 transactions               |
| Target     | `class` (1 = fraud, 0 = legitimate) |
| Fraud rate | ~10.6% (class imbalanced)           |
| Time span  | 2015                                |

**Feature categories:**

| Category     | Features                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------- |
| Temporal     | `signup_time`, `purchase_time` -> `time_to_purchase_secs`, `purchase_hour`, `purchase_day_of_week` |
| Transaction  | `purchase_value` (binned: Low / Medium / High / Very High)                                         |
| Identity     | `device_id`, `ip_address`, `user_id` (dropped — high cardinality ID)                               |
| Demographics | `age` (binned: Young / Adult / Middle Age / Senior), `sex`                                         |
| Behavioural  | `source`, `browser` (OHE encoded)                                                                  |
| Engineered   | Velocity:`device_txn_count_1h/24h`, `ip_amount_sum_1h/24h`, `device_rarity`, `ip_rarity`           |

---

## Model Performance

| Model               | AUC-ROC   | AUC-PR    | Precision | Recall    | F1        | Optimal Threshold |
| ------------------- | --------- | --------- | --------- | --------- | --------- | ----------------- |
| Logistic Regression | ~0.83     | ~0.51     | ~0.64     | ~0.62     | ~0.63     | ~0.18             |
| **LightGBM**        | **~0.93** | **~0.76** | **~0.78** | **~0.74** | **~0.76** | **~0.32**         |

> _Metrics are indicative - run `make train` locally for exact results on your data split._

**Cost savings vs. flag-all baseline** (LightGBM at optimal threshold):

- Flag-all cost (all FP): $135,900 (27,180 legit txns × $5)
- Model cost (FP + FN optimised): ~$18,400
- **Net savings: ~$117,500** per dataset replay

---

## Deployment Options

The spec evaluates three AWS ECS deployment strategies:

| Strategy                   | Best For                     | Cost             | Complexity |
| -------------------------- | ---------------------------- | ---------------- | ---------- |
| **Fargate**                | Bursty / unpredictable loads | Highest per-task | Lowest     |
| **EC2**                    | 24/7 steady-state workloads  | Lowest at scale  | Highest    |
| **Fargate + EC2 Hybrid** ✓ | Mixed batch + spike traffic  | Balanced         | Medium     |

**Recommended: Fargate + EC2 Hybrid** - EC2 reserved for Airflow batch DAGs, Fargate auto-scales for streaming spikes.

---

## Environment Variables

Copy `.env.example` -> `.env` and configure:

| Variable                     | Required      | Default                 | Description                          |
| ---------------------------- | ------------- | ----------------------- | ------------------------------------ |
| `MLFLOW_TRACKING_URI`        | ✓             | `http://localhost:5001` | MLflow server URI                    |
| `AWS_ACCESS_KEY`             | If using S3   | -                       | AWS IAM access key                   |
| `SECTRET_ACCESS_KEY`         | If using S3   | -                       | AWS IAM secret key                   |
| `S3_BUCKET`                  | If using S3   | -                       | MLflow artifact bucket name          |
| `KAFKA_BOOTSTRAP_SERVERS`    | For streaming | `localhost:9093`        | Kafka broker address                 |
| `STREAMING_DELAY_SECS`       | No            | `0.05`                  | Seconds between producer events      |
| `AIRFLOW_UID`                | If Linux      | `50000`                 | Host user UID for volume permissions |
| `_AIRFLOW_WWW_USER_PASSWORD` | For Airflow   | `admin`                 | Webserver admin password             |
| `RDS_HOST` / `RDS_PASSWORD`  | For Docker DB | `localhost`             | Postgres connection                  |

---

## Contributing

1. Fork the repository and create a feature branch
2. Make your changes with tests (`make test` must pass)
3. Run `make lint` - fix any ruff or black issues
4. Open a pull request - CI will run automatically
5. All 5 CI jobs must pass before merge

---

## License

MIT - see [LICENSE](LICENSE) for details.
