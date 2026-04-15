"""
DAG 3 — Batch Inference Pipeline
==================================
Loads the Production model from the MLflow registry, scores a batch of
transactions from the test set (or any CSV pointed to by INFERENCE_INPUT),
and exports predictions to artifacts/predictions/.

Steps:
  load_model  → score_batch  → export_scores

Schedule: Daily at 04:00 UTC (after training_dag has a chance to run overnight).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT        = os.environ.get("REPO_ROOT", "/opt/airflow")
MODEL_META_PATH  = os.path.join(REPO_ROOT, "artifacts", "models", "best_model_meta.json")
INFERENCE_INPUT  = os.environ.get(
    "INFERENCE_INPUT",
    os.path.join(REPO_ROOT, "artifacts", "data", "X_test.csv"),
)
PREDICTIONS_DIR  = os.path.join(REPO_ROOT, "artifacts", "predictions")

# ── Default args ───────────────────────────────────────────────────────────────
default_args = {
    "owner":            "ml_engineering",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 1, 1),
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# ── DAG ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="batch_inference_dag",
    description="Load Production model → Score batch → Export predictions CSV",
    schedule_interval="0 4 * * *",   # Daily at 04:00 UTC
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["inference", "batch", "fraud-detection"],
) as dag:

    # ── Task 1: load_model — verify model is available ─────────────────────────
    def _load_model(**context) -> dict:
        """
        Resolve the model URI to use for inference:
        1. Try to load 'Production' model from MLflow registry.
        2. Fall back to 'Staging' if no Production model exists.
        3. Fall back to local best_model_meta.json URI.
        Push model_info dict to XCom.
        """
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

        from mlflow_utils import setup_mlflow
        from config import get_mlflow_config
        import mlflow

        setup_mlflow()
        cfg          = get_mlflow_config()
        registry_name = cfg.get("model_registry_name", "fraud_detection")

        threshold = 0.5
        model_uri = None
        source    = None

        # Try Production → Staging → local meta
        for stage in ("Production", "Staging"):
            try:
                uri = f"models:/{registry_name}/{stage}"
                mlflow.pyfunc.load_model(uri)   # just validate it loads
                model_uri = uri
                source    = stage
                print(f"  ✓ Resolved model: {uri}")
                break
            except Exception as exc:
                print(f"  No {stage} model: {exc}")

        if model_uri is None and os.path.exists(MODEL_META_PATH):
            with open(MODEL_META_PATH) as fh:
                meta = json.load(fh)
            model_uri = meta.get("mlflow_uri")
            threshold = meta.get("threshold", 0.5)
            source    = "local_meta"
            print(f"  ✓ Resolved model from local meta: {model_uri}")

        if model_uri is None:
            raise RuntimeError(
                "No model found in MLflow registry or local artifacts. "
                "Run training_dag first."
            )

        model_info = {
            "model_uri": model_uri,
            "stage":     source,
            "threshold": threshold,
            "registry":  registry_name,
        }
        context["ti"].xcom_push(key="model_info", value=model_info)
        return model_info


    load_model = PythonOperator(
        task_id="load_model",
        python_callable=_load_model,
        doc_md="""
        **load_model**
        Resolves the model URI in order: `Production` → `Staging` → local meta JSON.
        Raises if no model is available. Pushes `model_info` dict to XCom.
        """,
    )

    # ── Task 2: score_batch — run inference on the input CSV ───────────────────
    def _score_batch(**context) -> dict:
        """
        Load the resolved model and score every row in INFERENCE_INPUT.
        Pushes a scored DataFrame (as records) to XCom.
        """
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

        import numpy as np
        import pandas as pd
        import mlflow

        model_info = context["ti"].xcom_pull(task_ids="load_model", key="model_info")
        model_uri  = model_info["model_uri"]
        threshold  = float(model_info.get("threshold", 0.5))

        if not os.path.exists(INFERENCE_INPUT):
            raise FileNotFoundError(f"Inference input not found: {INFERENCE_INPUT}")

        X = pd.read_csv(INFERENCE_INPUT)
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        print(f"  Scoring {len(X_num):,} rows with {len(X_num.columns)} features")
        print(f"  Model URI : {model_uri}")
        print(f"  Threshold : {threshold}")

        model  = mlflow.pyfunc.load_model(model_uri)
        raw    = model.predict(X_num)

        # raw may be class predictions (0/1) or probabilities
        if raw.max() <= 1.0 and raw.dtype in (float, "float64", "float32"):
            proba = raw
        else:
            proba = raw.astype(float)

        decisions = (proba >= threshold).astype(int)

        results = X.copy()
        results["fraud_probability"] = proba.round(6)
        results["fraud_decision"]    = decisions
        results["threshold_used"]    = threshold
        results["scored_at"]         = datetime.utcnow().isoformat()
        results["model_uri"]         = model_uri

        stats = {
            "total_scored":   len(results),
            "fraud_flagged":  int(decisions.sum()),
            "fraud_rate":     round(float(decisions.mean()), 4),
        }
        print(f"\n  Scored {stats['total_scored']:,} rows")
        print(f"  Fraud flagged : {stats['fraud_flagged']:,}  ({stats['fraud_rate']:.2%})")

        context["ti"].xcom_push(key="score_stats", value=stats)
        # Store as JSON-serialisable records in XCom (truncated to 500 rows for safety)
        context["ti"].xcom_push(key="scored_sample", value=results.head(500).to_dict("records"))
        return stats


    score_batch = PythonOperator(
        task_id="score_batch",
        python_callable=_score_batch,
        doc_md="""
        **score_batch**
        Loads the model from the resolved URI, scores every row in the input
        CSV, applies the cost-optimised threshold, and pushes scored results
        and summary statistics to XCom.
        """,
    )

    # ── Task 3: export_scores — write predictions CSV ──────────────────────────
    def _export_scores(**context) -> str:
        """
        Pull scored results from XCom and write a timestamped predictions CSV
        to artifacts/predictions/. Also writes a JSON summary sidecar.
        """
        import pandas as pd

        scored_sample = context["ti"].xcom_pull(task_ids="score_batch", key="scored_sample")
        score_stats   = context["ti"].xcom_pull(task_ids="score_batch", key="score_stats")
        model_info    = context["ti"].xcom_pull(task_ids="load_model",  key="model_info")

        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        ts_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        # Predictions CSV
        csv_path = os.path.join(PREDICTIONS_DIR, f"predictions_{ts_tag}.csv")
        df       = pd.DataFrame(scored_sample)
        df.to_csv(csv_path, index=False)

        # JSON summary sidecar
        summary = {
            "predictions_file": csv_path,
            "inference_input":  INFERENCE_INPUT,
            "model_uri":        model_info["model_uri"],
            "model_stage":      model_info["stage"],
            "threshold_used":   model_info["threshold"],
            "total_scored":     score_stats["total_scored"],
            "fraud_flagged":    score_stats["fraud_flagged"],
            "fraud_rate":       score_stats["fraud_rate"],
            "exported_at":      datetime.utcnow().isoformat(),
        }
        summary_path = os.path.join(PREDICTIONS_DIR, f"summary_{ts_tag}.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"\n  ✓ Predictions → {csv_path}")
        print(f"  ✓ Summary     → {summary_path}")
        print(f"  Total scored  : {score_stats['total_scored']:,}")
        print(f"  Fraud flagged : {score_stats['fraud_flagged']:,} ({score_stats['fraud_rate']:.2%})")

        return csv_path


    export_scores = PythonOperator(
        task_id="export_scores",
        python_callable=_export_scores,
        doc_md="""
        **export_scores**
        Writes the scored predictions to a timestamped CSV in
        `artifacts/predictions/` and saves a JSON summary sidecar for
        audit/lineage tracking.
        """,
    )

    # ── Dependencies ───────────────────────────────────────────────────────────
    load_model >> score_batch >> export_scores
