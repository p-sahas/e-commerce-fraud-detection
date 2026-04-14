"""
MLflow utility module for e-commerce fraud detection.

Provides a single consistent interface for all experiment tracking needs:
  - setup_mlflow()            Configure tracking URI and return the experiment
  - run()                     Context-manager that wraps mlflow.start_run()
  - log_params()              Batch-safe param logging (handles 250-char limit)
  - log_metrics_dict()        Log a dict of metrics in one call
  - log_data_quality()        Log a data quality report dict as params + artifact
  - promote_model()           Transition a model version through registry stages
  - get_production_model()    Load the current Production model from registry
  - get_latest_run()          Fetch the latest run for an experiment
  - compare_runs()            Return a DataFrame summary of all runs in experiment
  - print_run_summary()       Pretty-print a run's metrics to stdout
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from config import get_mlflow_config  # noqa: E402


# =============================================================================
# Internal helpers
# =============================================================================

def _load_cfg() -> Dict[str, Any]:
    """Return the mlflow config block, preferring MLFLOW_TRACKING_URI env var."""
    cfg = get_mlflow_config()
    # Env var overrides config file (important for Docker / CI environments)
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        cfg["tracking_uri"] = env_uri
    return cfg


# =============================================================================
# Experiment Setup
# =============================================================================

def setup_mlflow(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> mlflow.entities.Experiment:
    """
    Configure MLflow tracking URI and create / retrieve the experiment.

    Precedence for tracking URI:
        1. ``tracking_uri`` argument
        2. ``MLFLOW_TRACKING_URI`` environment variable
        3. ``mlflow.tracking_uri`` in config.yaml
        4. Local fallback: ``file:./mlruns``

    Args:
        experiment_name: Override experiment name from config.
        tracking_uri:    Override tracking URI.

    Returns:
        The MLflow Experiment object.
    """
    cfg = _load_cfg()

    uri  = tracking_uri or cfg.get("tracking_uri", "file:./mlruns")
    name = experiment_name or cfg.get("experiment_name", "E-Commerce Fraud Detection")
    tags = cfg.get("tags", {})

    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI : {uri}")

    # Set autologging if enabled in config
    if cfg.get("autolog", False):
        mlflow.autolog(silent=True)

    # get_or_create experiment
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name, tags=tags)
        exp    = client.get_experiment(exp_id)
        logger.info(f"Created MLflow experiment '{name}'  id={exp_id}")
    else:
        logger.info(f"Using MLflow experiment '{name}'  id={exp.experiment_id}")

    mlflow.set_experiment(name)
    return exp


# =============================================================================
# Run Context Manager
# =============================================================================

@contextmanager
def run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
    description: Optional[str] = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """
    Context manager wrapper around ``mlflow.start_run()`` that adds:
      - Automatic run name from config prefix if not given
      - UTC timestamp tag on every run
      - Safe cleanup even on exceptions

    Usage::

        with mlflow_utils.run(run_name="lgbm_v3") as active_run:
            mlflow.log_param("lr", 0.05)
            ...

    Args:
        run_name:    Human-readable run name.
        tags:        Extra tags merged with config-level tags.
        nested:      Start a nested child run (for hyperparameter sweeps).
        description: Optional run description stored as a tag.

    Yields:
        The active MLflow run object.
    """
    cfg = _load_cfg()

    # Build run name
    prefix   = cfg.get("run_name_prefix", "fd_run")
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    name     = run_name or f"{prefix}_{ts}"

    # Merge tags
    base_tags = dict(cfg.get("tags", {}))
    base_tags["mlflow.runName"] = name
    base_tags["started_at_utc"] = ts
    if description:
        base_tags["mlflow.note.content"] = description
    if tags:
        base_tags.update(tags)

    with mlflow.start_run(run_name=name, tags=base_tags, nested=nested) as active_run:
        logger.info(f"MLflow run started: '{name}'  id={active_run.info.run_id}")
        try:
            yield active_run
        except Exception as exc:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.set_tag("failure_reason", str(exc)[:500])
            logger.error(f"MLflow run '{name}' failed: {exc}")
            raise
        else:
            mlflow.set_tag("run_status", "SUCCESS")

    logger.info(f"MLflow run finished: '{name}'  id={active_run.info.run_id}")


# =============================================================================
# Logging helpers
# =============================================================================

def log_params(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Log a dict of parameters, safely handling MLflow's 250-char value limit
    and prefixing every key with ``prefix`` if given.

    Args:
        params: Dict of param_name → value.
        prefix: Optional prefix added to every key (e.g. ``"lgbm_"``).
    """
    safe = {}
    for k, v in params.items():
        key = f"{prefix}{k}" if prefix else k
        val = str(v)[:250]          # MLflow hard limit on param value length
        safe[key] = val

    # Log in batches of 100 (MLflow API limit per call)
    items = list(safe.items())
    for i in range(0, len(items), 100):
        batch = dict(items[i : i + 100])
        mlflow.log_params(batch)


def log_metrics_dict(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log a dict of metrics in one call, skipping non-numeric values.

    Args:
        metrics: Dict of metric_name → numeric value.
        step:    Optional step/epoch number.
    """
    numeric = {}
    for k, v in metrics.items():
        try:
            numeric[k] = float(v)
        except (TypeError, ValueError):
            logger.debug(f"Skipping non-numeric metric '{k}': {v!r}")

    if numeric:
        mlflow.log_metrics(numeric, step=step)


def log_data_quality(report: Dict[str, Any], artifact_filename: str = "data_quality_report.json") -> None:
    """
    Log a data quality report dict — flat numeric fields become params,
    the full report is saved as a JSON artifact.

    Args:
        report:            Dict produced by the data pipeline quality check.
        artifact_filename: Name of the JSON file saved under ``/data_quality/``.
    """
    # Log scalar fields as params for easy filtering in the UI
    for k, v in report.items():
        if isinstance(v, (int, float, bool, str)):
            try:
                mlflow.log_param(f"dq_{k}", str(v)[:250])
            except Exception:
                pass

    # Save full report as artifact
    os.makedirs("artifacts/reports", exist_ok=True)
    local_path = os.path.join("artifacts", "reports", artifact_filename)
    with open(local_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    mlflow.log_artifact(local_path, artifact_path="data_quality")
    logger.info(f"Data quality report logged → {local_path}")


# =============================================================================
# Model Registry
# =============================================================================

def promote_model(
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    stage: str = "Staging",
    archive_existing: bool = True,
    comment: Optional[str] = None,
) -> None:
    """
    Transition a registered model version to the target stage.

    Stages: ``"None"`` → ``"Staging"`` → ``"Production"`` → ``"Archived"``

    Args:
        model_name:       Registry name (defaults to config ``model_registry_name``).
        version:          Version string, e.g. ``"3"``. Defaults to the latest version.
        stage:            Target stage.
        archive_existing: Archive current models in ``stage`` before promoting.
        comment:          Optional annotation stored on the transition.
    """
    cfg    = _load_cfg()
    name   = model_name or cfg.get("model_registry_name", "fraud_detection")
    client = MlflowClient()

    if version is None:
        # Pick the latest version regardless of current stage
        versions = client.get_latest_versions(name)
        if not versions:
            raise ValueError(f"No versions found for model '{name}'")
        version = str(max(int(v.version) for v in versions))

    if archive_existing:
        for mv in client.get_latest_versions(name, stages=[stage]):
            client.transition_model_version_stage(
                name=name, version=mv.version, stage="Archived"
            )
            logger.info(f"  Archived '{name}' v{mv.version} from {stage}")

    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage,
        archive_existing_versions=False,
    )

    if comment:
        client.update_model_version(name=name, version=version, description=comment)

    logger.info(f"  Promoted '{name}' v{version} → {stage}")


def get_production_model(model_name: Optional[str] = None) -> mlflow.pyfunc.PyFuncModel:
    """
    Load the current ``Production`` model from the MLflow registry.

    Args:
        model_name: Registry name (defaults to config ``model_registry_name``).

    Returns:
        Loaded ``mlflow.pyfunc.PyFuncModel``.

    Raises:
        ValueError: If no Production model is registered.
    """
    cfg  = _load_cfg()
    name = model_name or cfg.get("model_registry_name", "fraud_detection")
    uri  = f"models:/{name}/Production"
    try:
        model = mlflow.pyfunc.load_model(uri)
        logger.info(f"Loaded Production model '{name}' from {uri}")
        return model
    except Exception as exc:
        raise ValueError(
            f"No Production model found for '{name}'. "
            "Run 'make mlflow-promote' to promote a Staging model first."
        ) from exc


def get_latest_model(model_name: Optional[str] = None, stage: str = "latest") -> mlflow.pyfunc.PyFuncModel:
    """
    Load the latest model version from the registry (any stage).

    Args:
        model_name: Registry name.
        stage:      Stage filter — ``"latest"``, ``"Staging"``, ``"Production"``.

    Returns:
        Loaded ``mlflow.pyfunc.PyFuncModel``.
    """
    cfg  = _load_cfg()
    name = model_name or cfg.get("model_registry_name", "fraud_detection")
    uri  = f"models:/{name}/{stage}"
    model = mlflow.pyfunc.load_model(uri)
    logger.info(f"Loaded model '{name}' ({stage}) from {uri}")
    return model


# =============================================================================
# Run Inspection
# =============================================================================

def get_latest_run(experiment_name: Optional[str] = None) -> Optional[Run]:
    """
    Return the most-recently completed run for the given experiment.

    Args:
        experiment_name: Defaults to config ``experiment_name``.

    Returns:
        MLflow ``Run`` object or ``None`` if no runs exist.
    """
    cfg    = _load_cfg()
    name   = experiment_name or cfg.get("experiment_name", "E-Commerce Fraud Detection")
    client = MlflowClient()
    exp    = client.get_experiment_by_name(name)
    if exp is None:
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def compare_runs(
    experiment_name: Optional[str] = None,
    metric_keys: Optional[List[str]] = None,
    max_results: int = 20,
) -> "pd.DataFrame":
    """
    Return a pandas DataFrame summarising recent runs for easy comparison.

    Columns: run_id, run_name, status, start_time, model_type,
             auc_roc, auc_pr, f1_score, precision, recall, threshold

    Args:
        experiment_name: Defaults to config experiment name.
        metric_keys:     Metrics to include (defaults to key fraud metrics).
        max_results:     Maximum number of runs to return.

    Returns:
        pandas DataFrame, sorted by auc_roc descending.
    """
    import pandas as pd  # lazy import — not required in all contexts

    cfg    = _load_cfg()
    name   = experiment_name or cfg.get("experiment_name", "E-Commerce Fraud Detection")
    client = MlflowClient()
    exp    = client.get_experiment_by_name(name)
    if exp is None:
        logger.warning(f"Experiment '{name}' not found.")
        return pd.DataFrame()

    if metric_keys is None:
        metric_keys = ["auc_roc", "auc_pr", "f1_score", "precision", "recall",
                       "threshold", "cost_savings", "expected_cost"]

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=max_results,
    )

    rows = []
    for r in runs:
        row: Dict[str, Any] = {
            "run_id":     r.info.run_id,
            "run_name":   r.data.tags.get("mlflow.runName", ""),
            "status":     r.info.status,
            "start_time": datetime.fromtimestamp(
                r.info.start_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "model_type": r.data.params.get("model_type", ""),
        }
        for key in metric_keys:
            row[key] = r.data.metrics.get(key)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "auc_roc" in df.columns:
        df = df.sort_values("auc_roc", ascending=False)
    return df


def print_run_summary(run_id: Optional[str] = None) -> None:
    """
    Pretty-print the metrics and params of a run to stdout.

    Args:
        run_id: Specific run ID, or the latest run if ``None``.
    """
    client = MlflowClient()

    if run_id is None:
        r = get_latest_run()
        if r is None:
            print("No runs found.")
            return
    else:
        r = client.get_run(run_id)

    width = 60
    print(f"\n{'═' * width}")
    print(f"  Run: {r.data.tags.get('mlflow.runName', r.info.run_id)}")
    print(f"  ID : {r.info.run_id}")
    print(f"  Status : {r.info.status}")
    print(f"{'─' * width}")

    key_metrics = ["auc_roc", "auc_pr", "f1_score", "precision", "recall",
                   "threshold", "cost_savings", "expected_cost"]
    print("  Metrics:")
    for k in key_metrics:
        v = r.data.metrics.get(k)
        if v is not None:
            print(f"    {k:<25} {v:.4f}")

    print(f"{'─' * width}")
    print("  Params:")
    for k, v in sorted(r.data.params.items()):
        print(f"    {k:<30} {v}")
    print(f"{'═' * width}\n")


# =============================================================================
# CLI helper — quick status check
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow utilities CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("summary",  help="Print latest run summary")
    sub.add_parser("compare",  help="Print run comparison table")

    p_promote = sub.add_parser("promote", help="Promote a model to a registry stage")
    p_promote.add_argument("--stage",   default="Production", help="Target stage")
    p_promote.add_argument("--version", default=None,         help="Version to promote")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup_mlflow()

    if args.cmd == "summary":
        print_run_summary()

    elif args.cmd == "compare":
        import pandas as pd
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(compare_runs())

    elif args.cmd == "promote":
        promote_model(stage=args.stage, version=args.version)
        print(f"Model promoted to '{args.stage}'.")

    else:
        parser.print_help()
