"""
DAG 1 - Data Pipeline
======================
Orchestrates the full data ingestion and preprocessing workflow.

Steps:
  ingest          -> validate schema      -> quality_report
  -> curated_export (saves processed CSVs to artifacts/data/)

Schedule: Daily at 02:00 UTC
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

# Paths
REPO_ROOT   = os.environ.get("REPO_ROOT", "/opt/airflow")
RAW_DATA    = os.path.join(REPO_ROOT, "data", "raw", "Fraud_Data.csv")
CURATED_DIR = os.path.join(REPO_ROOT, "artifacts", "data")
REPORT_PATH = os.path.join(REPO_ROOT, "artifacts", "reports", "data_quality_report.json")

# Default args
default_args = {
    "owner":            "ml_engineering",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 1, 1),
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# DAG
with DAG(
    dag_id="data_pipeline_dag",
    description="Ingest -> Validate -> Quality Report -> Curated Export",
    schedule_interval="0 2 * * *",   # daily at 02:00 UTC
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["data", "fraud-detection", "etl"],
) as dag:

    # Task 1: Wait for raw data file
    wait_for_raw_data = FileSensor(
        task_id="wait_for_raw_data",
        filepath=RAW_DATA,
        poke_interval=30,
        timeout=600,
        mode="poke",
        doc_md="""
        **wait_for_raw_data**
        Polls until `data/raw/Fraud_Data.csv` is present. Blocks the rest of the
        pipeline until the file exists, preventing processing of missing data.
        """,
    )

    # Task 2: Schema validation
    def _schema_validate(**context) -> dict:
        """
        Validate raw CSV against the expected schema:
        - Required columns present
        - Column types match expectations
        - No duplicate user_id + purchase_time combinations
        Push a validation report dict to XCom.
        """
        import pandas as pd

        required_cols = [
            "user_id", "signup_time", "purchase_time", "purchase_value",
            "device_id", "source", "browser", "sex", "age", "ip_address", "class",
        ]
        numeric_cols = ["purchase_value", "age", "ip_address"]
        string_cols  = ["source", "browser", "sex"]

        df = pd.read_csv(RAW_DATA)

        errors  = []
        warnings = []

        # Check required columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

        # Check numeric types
        for col in numeric_cols:
            if col in df.columns:
                non_numeric = df[col].apply(lambda x: not str(x).replace(".", "").replace("-", "").isdigit()).sum()
                if non_numeric > 0:
                    warnings.append(f"Column '{col}' has {non_numeric} non-numeric values")

        # Check temporal ordering: signup_time <= purchase_time
        if "signup_time" in df.columns and "purchase_time" in df.columns:
            df["signup_time"]   = pd.to_datetime(df["signup_time"],   errors="coerce")
            df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
            bad_temporal = (df["purchase_time"] < df["signup_time"]).sum()
            if bad_temporal > 0:
                errors.append(f"Temporal violation: {bad_temporal} rows where purchase_time < signup_time")

        # Target column check
        if "class" in df.columns:
            invalid_labels = (~df["class"].isin([0, 1])).sum()
            if invalid_labels > 0:
                errors.append(f"Target 'class' contains {invalid_labels} non-binary values")

        report = {
            "total_rows":  len(df),
            "total_cols":  len(df.columns),
            "columns":     list(df.columns),
            "errors":      errors,
            "warnings":    warnings,
            "passed":      len(errors) == 0,
            "validated_at": datetime.utcnow().isoformat(),
        }

        if errors:
            raise ValueError(f"Schema validation FAILED: {errors}")

        context["ti"].xcom_push(key="schema_report", value=report)
        return report


    schema_validate = PythonOperator(
        task_id="schema_validate",
        python_callable=_schema_validate,
        doc_md="""
        **schema_validate**
        Enforces field types, required columns, value ranges, and temporal ordering.
        Raises immediately on hard errors; logs warnings for soft anomalies.
        Pushes the validation report to XCom for downstream tasks.
        """,
    )

    # Task 3: Quality report
    def _quality_report(**context) -> dict:
        """
        Compute a comprehensive data quality report:
        - Row / fraud counts and ratio
        - Missing value counts per column
        - Outlier detection (IQR-based) for purchase_value and age
        - Time span of transaction data
        Save report to artifacts/reports/data_quality_report.json.
        """
        import pandas as pd
        import numpy as np

        df = pd.read_csv(RAW_DATA)
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
        df["signup_time"]   = pd.to_datetime(df["signup_time"],   errors="coerce")

        total_rows  = len(df)
        fraud_count = int(df["class"].sum()) if "class" in df.columns else 0
        fraud_ratio = round(fraud_count / max(total_rows, 1), 4)

        # Missing values
        missing = {col: int(df[col].isna().sum()) for col in df.columns}
        total_missing = sum(missing.values())

        # Outliers via IQR for numeric cols
        outlier_counts = {}
        for col in ["purchase_value", "age"]:
            if col in df.columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr    = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())

        # Time span
        time_span_days = None
        if "purchase_time" in df.columns:
            ts_sorted = df["purchase_time"].dropna().sort_values()
            if len(ts_sorted) > 1:
                time_span_days = (ts_sorted.iloc[-1] - ts_sorted.iloc[0]).days

        report = {
            "total_rows":       total_rows,
            "total_columns":    len(df.columns),
            "fraud_count":      fraud_count,
            "legit_count":      total_rows - fraud_count,
            "fraud_ratio":      fraud_ratio,
            "total_missing":    total_missing,
            "missing_by_col":   missing,
            "outliers_by_col":  outlier_counts,
            "time_span_days":   time_span_days,
            "generated_at":     datetime.utcnow().isoformat(),
        }

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w") as fh:
            json.dump(report, fh, indent=2)

        context["ti"].xcom_push(key="quality_report", value=report)
        print(f"Quality report saved -> {REPORT_PATH}")
        print(f"  Rows: {total_rows:,}  |  Fraud ratio: {fraud_ratio:.2%}")
        print(f"  Missing values: {total_missing:,}  |  Outliers: {outlier_counts}")
        return report


    quality_report = PythonOperator(
        task_id="quality_report",
        python_callable=_quality_report,
        doc_md="""
        **quality_report**
        Produces a JSON quality report covering row counts, fraud ratio, missing
        values per column, IQR-based outlier counts, and transaction time span.
        """,
    )

    # Task 4: Curated export (run PySpark data pipeline)
    curated_export = BashOperator(
        task_id="curated_export",
        bash_command=(
            f"cd {REPO_ROOT} && "
            "python pipelines/data_pipeline.py"
        ),
        doc_md="""
        **curated_export**
        Runs the full PySpark data pipeline. Outputs:
        - artifacts/data/X_train.csv  +  X_train.parquet
        - artifacts/data/X_test.csv   +  X_test.parquet
        - artifacts/data/Y_train.csv  +  Y_train.parquet
        - artifacts/data/Y_test.csv   +  Y_test.parquet
        """,
    )

    # Dependencies
    wait_for_raw_data >> schema_validate >> quality_report >> curated_export
