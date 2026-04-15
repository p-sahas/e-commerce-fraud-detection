"""
DAG 2 — Training Pipeline
==========================
Orchestrates the end-to-end model training workflow triggered after the
data pipeline DAG succeeds.

Steps:
  load_curated  → feature_build  → kfold_train (LR + LightGBM)
  → eval        → register_model

Schedule: Triggered externally after data_pipeline_dag, or weekly on Monday 03:00 UTC.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.trigger_rule import TriggerRule

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.environ.get("REPO_ROOT", "/opt/airflow")
ARTIFACTS_DIR  = os.path.join(REPO_ROOT, "artifacts", "data")
MODEL_META     = os.path.join(REPO_ROOT, "artifacts", "models", "best_model_meta.json")
X_TRAIN_PATH   = os.path.join(ARTIFACTS_DIR, "X_train.csv")

# ── Default args ───────────────────────────────────────────────────────────────
default_args = {
    "owner":            "ml_engineering",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 1, 1),
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# ── DAG ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="training_dag",
    description="Load curated data → Feature build → K-Fold train → Eval → Register model",
    schedule_interval="0 3 * * 1",   # Weekly Monday 03:00 UTC; trigger manually after data DAG
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlflow", "fraud-detection"],
) as dag:

    # ── Task 1: Wait for data_pipeline_dag curated export ─────────────────────
    wait_for_data_dag = ExternalTaskSensor(
        task_id="wait_for_data_dag",
        external_dag_id="data_pipeline_dag",
        external_task_id="curated_export",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
        doc_md="""
        **wait_for_data_dag**
        Waits for the `curated_export` task in `data_pipeline_dag` to succeed
        before starting training. Uses `reschedule` mode to free a worker slot
        while waiting.
        """,
    )

    # ── Task 2: load_curated — verify artifacts exist and log stats ────────────
    def _load_curated(**context) -> dict:
        """
        Verify all four curated CSVs exist and log basic statistics.
        Pushes dataset info to XCom for downstream tasks.
        """
        import pandas as pd

        files = {
            "X_train": os.path.join(ARTIFACTS_DIR, "X_train.csv"),
            "X_test":  os.path.join(ARTIFACTS_DIR, "X_test.csv"),
            "Y_train": os.path.join(ARTIFACTS_DIR, "Y_train.csv"),
            "Y_test":  os.path.join(ARTIFACTS_DIR, "Y_test.csv"),
        }

        missing = [k for k, v in files.items() if not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(f"Missing curated files: {missing}. Run data_pipeline_dag first.")

        X_train = pd.read_csv(files["X_train"])
        Y_train = pd.read_csv(files["Y_train"])

        stats = {
            "n_train":       len(X_train),
            "n_features":    len(X_train.columns),
            "n_test":        len(pd.read_csv(files["X_test"])),
            "fraud_ratio":   round(Y_train.iloc[:, 0].mean(), 4),
            "feature_cols":  list(X_train.columns),
        }

        print(f"  Training set : {stats['n_train']:,} rows  |  {stats['n_features']} features")
        print(f"  Fraud ratio  : {stats['fraud_ratio']:.2%}")

        context["ti"].xcom_push(key="dataset_stats", value=stats)
        return stats


    load_curated = PythonOperator(
        task_id="load_curated",
        python_callable=_load_curated,
        doc_md="""
        **load_curated**
        Verified the four CSV artifacts produced by the data pipeline exist
        and pushes dataset statistics to XCom.
        """,
    )

    # ── Task 3: feature_build — report feature statistics ─────────────────────
    def _feature_build(**context) -> dict:
        """
        Compute and log feature statistics — correlation with target,
        variance, null rates — to help trace feature drift over time.
        """
        import pandas as pd
        import numpy as np

        X = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train.csv"))
        y = pd.read_csv(os.path.join(ARTIFACTS_DIR, "Y_train.csv")).iloc[:, 0]

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        corr     = X[num_cols].corrwith(y).abs().sort_values(ascending=False)

        feat_report = {
            "top_5_correlated":  corr.head(5).to_dict(),
            "zero_variance_cols": [c for c in X.columns if X[c].nunique() <= 1],
            "high_null_cols":     [c for c in X.columns if X[c].isna().mean() > 0.05],
            "n_numeric_features": len(num_cols),
        }

        print(f"  Top correlated features: {list(corr.head(5).index)}")
        if feat_report["zero_variance_cols"]:
            print(f"  ⚠ Zero-variance columns: {feat_report['zero_variance_cols']}")

        context["ti"].xcom_push(key="feature_report", value=feat_report)
        return feat_report


    feature_build = PythonOperator(
        task_id="feature_build",
        python_callable=_feature_build,
        doc_md="""
        **feature_build**
        Computes per-feature statistics (correlation with target, variance, null
        rate) and flags zero-variance or high-null columns.
        """,
    )

    # ── Task 4: kfold_train — run the full training pipeline ──────────────────
    kfold_train = BashOperator(
        task_id="kfold_train",
        bash_command=(
            f"cd {REPO_ROOT} && "
            "python pipelines/training_pipeline.py "
            "--models logistic_regression lightgbm "
            "--min-auc 0.75"
        ),
        doc_md="""
        **kfold_train**
        Runs `training_pipeline.py` which trains both Logistic Regression and
        LightGBM with TimeSeriesSplit cross-validation, optimises the decision
        threshold, and logs every metric and artifact to MLflow.
        """,
    )

    # ── Task 5: eval — check AUC gate and emit report ─────────────────────────
    def _eval(**context) -> dict:
        """
        Read the best_model_meta.json produced by the training pipeline and
        verify the AUC gate, then push the metrics to XCom.
        """
        if not os.path.exists(MODEL_META):
            raise FileNotFoundError(
                f"best_model_meta.json not found at {MODEL_META}. "
                "Did kfold_train succeed?"
            )

        with open(MODEL_META) as fh:
            meta = json.load(fh)

        metrics   = meta.get("metrics", {})
        auc_roc   = metrics.get("auc_roc", 0.0)
        min_gate  = float(os.environ.get("MIN_AUC_GATE", "0.75"))

        print(f"\n  Best model   : {meta.get('model_type')}")
        print(f"  Threshold    : {meta.get('threshold'):.2f}")
        print(f"  AUC-ROC      : {auc_roc:.4f}  (gate: {min_gate:.2f})")
        print(f"  F1 score     : {metrics.get('f1_score', 'n/a')}")
        print(f"  Cost savings : ${metrics.get('cost_savings', 0):,.0f}")

        if auc_roc < min_gate:
            raise ValueError(
                f"AUC gate FAILED: {auc_roc:.4f} < {min_gate:.2f}. "
                "Model will NOT be registered. Check feature quality or training config."
            )

        context["ti"].xcom_push(key="model_meta", value=meta)
        return meta


    eval_task = PythonOperator(
        task_id="eval",
        python_callable=_eval,
        doc_md="""
        **eval**
        Reads `best_model_meta.json`, enforces the AUC gate (default 0.75),
        and pushes model metadata to XCom for the registration step.
        Raises an error to halt the DAG if the gate fails.
        """,
    )

    # ── Task 6: register_model — promote best model to Staging in MLflow ───────
    def _register_model(**context) -> None:
        """
        Use mlflow_utils to promote the latest registered version to Staging.
        The model was already logged to the registry by training_pipeline.py;
        this step handles the stage transition.
        """
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

        from mlflow_utils import setup_mlflow, promote_model

        setup_mlflow()
        promote_model(stage="Staging", archive_existing=True,
                      comment=f"Auto-promoted by training_dag on {datetime.utcnow().date()}")
        print("  Model promoted to Staging in MLflow registry.")


    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
        doc_md="""
        **register_model**
        Transitions the latest model version in the MLflow registry to `Staging`.
        Manual approval in the MLflow UI is required to move to `Production`.
        """,
    )

    # ── Dependencies ───────────────────────────────────────────────────────────
    (
        wait_for_data_dag
        >> load_curated
        >> feature_build
        >> kfold_train
        >> eval_task
        >> register_model
    )
