"""
conftest.py — Shared pytest fixtures for the fraud detection test suite.

Fixtures available:
  sample_df_pd        — pandas DataFrame with realistic transaction columns
  sample_X_y          — (X, y) numpy arrays for model/threshold tests
  cost_matrix         — standard business cost dict (fp=$5, fn=$100)
  streaming_events    — list of raw event dicts matching the Kafka message schema
  tmp_mlflow_dir      — temporary local MLflow tracking directory (autouse per session)
"""

import json
import os
import sys
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# Path bootstrap
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipelines"))


# =============================================================================
# MLflow — session-scoped temporary tracking dir
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def tmp_mlflow_dir(tmp_path_factory) -> str:
    """
    Set MLFLOW_TRACKING_URI to a temp local dir for the entire test session.
    Prevents tests from accidentally writing to the real mlruns/ directory.
    """
    tracking_dir = str(tmp_path_factory.mktemp("mlruns"))
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{tracking_dir}"

    import mlflow
    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment("test_experiment")

    return tracking_dir


# =============================================================================
# Tabular data
# =============================================================================

@pytest.fixture
def sample_df_pd() -> pd.DataFrame:
    """
    Minimal pandas DataFrame resembling the raw Fraud_Data.csv schema.
    100 rows, ~10% fraud, no missing values.
    """
    rng = np.random.default_rng(42)
    n   = 100

    signup  = pd.date_range("2022-01-01", periods=n, freq="6h")
    purchase = signup + pd.to_timedelta(rng.integers(60, 86_400, size=n), unit="s")

    df = pd.DataFrame({
        "user_id":        rng.integers(1_000, 9_999, size=n).astype(str),
        "signup_time":    signup.strftime("%Y-%m-%d %H:%M:%S"),
        "purchase_time":  purchase.strftime("%Y-%m-%d %H:%M:%S"),
        "purchase_value": rng.uniform(5.0, 500.0, size=n).round(2),
        "device_id":      [f"D{rng.integers(100,999)}" for _ in range(n)],
        "source":         rng.choice(["SEO", "Ads", "Direct"], size=n),
        "browser":        rng.choice(["Chrome", "Firefox", "Safari"], size=n),
        "sex":            rng.choice(["M", "F"], size=n),
        "age":            rng.integers(18, 70, size=n),
        "ip_address":     rng.uniform(1e8, 3e9, size=n).astype(int),
        "class":          rng.choice([0, 1], size=n, p=[0.90, 0.10]),
    })
    return df


@pytest.fixture
def sample_df_with_nulls(sample_df_pd) -> pd.DataFrame:
    """sample_df_pd with deliberate nulls in purchase_value and source."""
    df = sample_df_pd.copy()
    idx = [3, 17, 42]
    df.loc[idx, "purchase_value"] = np.nan
    df.loc[idx, "source"]         = None
    return df


@pytest.fixture
def sample_df_duplicates(sample_df_pd) -> pd.DataFrame:
    """sample_df_pd with 5 duplicated rows appended."""
    return pd.concat([sample_df_pd, sample_df_pd.iloc[:5]], ignore_index=True)


@pytest.fixture
def sample_X_y() -> Tuple[np.ndarray, np.ndarray]:
    """
    (X, y) arrays with 200 samples, 8 numeric features, ~10% fraud.
    Suitable for training a trivial sklearn model in tests.
    """
    rng = np.random.default_rng(0)
    n   = 200
    X   = rng.standard_normal((n, 8)).astype(np.float32)
    # Slightly separate the classes to make AUC > 0.5
    y   = (X[:, 0] + rng.standard_normal(n) * 0.8 > 0.5).astype(int)
    return X, y


@pytest.fixture
def cost_matrix() -> dict:
    """Standard business cost parameters from E2E spec."""
    return {"fp_cost": 5.0, "fn_cost": 100.0}


# =============================================================================
# Streaming events
# =============================================================================

@pytest.fixture
def streaming_events() -> list:
    """
    10 synthetic transaction events matching the Kafka message JSON schema
    used by the streaming pipeline producer.
    """
    rng = np.random.default_rng(7)
    base_ts = 1_700_000_000.0   # ~Nov 2023

    events = []
    for i in range(10):
        events.append({
            "user_id":             str(1000 + i),
            "device_id":           f"DEV{100 + (i % 3)}",
            "ip_address":          float(int(rng.uniform(1e8, 3e9))),
            "purchase_value":      round(float(rng.uniform(10.0, 300.0)), 2),
            "source":              rng.choice(["SEO", "Ads", "Direct"]).item(),
            "browser":             rng.choice(["Chrome", "Firefox"]).item(),
            "sex":                 rng.choice(["M", "F"]).item(),
            "age":                 int(rng.integers(20, 60)),
            "purchase_time_ts":    base_ts + i * 120,    # 2 minutes apart
            "class":               int(rng.choice([0, 1], p=[0.9, 0.1])),
        })
    return events


# =============================================================================
# Temporary model meta JSON
# =============================================================================

@pytest.fixture
def model_meta_json(tmp_path) -> str:
    """Write a minimal best_model_meta.json to a temp path and return the path."""
    meta = {
        "model_type": "lightgbm",
        "threshold":  0.42,
        "registry":   "fraud_detection",
        "mlflow_uri": "runs:/abc123/model",
        "run_id":     "abc123",
        "metrics": {
            "auc_roc":   0.92,
            "auc_pr":    0.78,
            "f1_score":  0.64,
            "precision": 0.71,
            "recall":    0.58,
        },
    }
    fpath = tmp_path / "best_model_meta.json"
    fpath.write_text(json.dumps(meta))
    return str(fpath)
