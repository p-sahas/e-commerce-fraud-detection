"""
Training pipeline orchestrator for e-commerce fraud detection.

Wires together:
  1. Loading preprocessed CSV artifacts from the data pipeline
  2. Training Logistic Regression (baseline) with MLflow tracking
  3. Training LightGBM (advanced)     with MLflow tracking
  4. Cost-aware threshold optimisation on a held-out slice of training data
  5. Full evaluation on the held-out test set
  6. AUC quality gate (≥ 0.75 required)
  7. Registration of the best model in the MLflow Model Registry

Usage:
    python pipelines/training_pipeline.py

Or via Makefile:
    make train
"""

import os
import sys
import logging
from typing import Dict, List, Optional

import numpy as np
import mlflow

# Path bootstrap ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from training import (
    load_processed_data,
    train_logistic_regression,
    train_lightgbm,
    optimize_threshold,
    evaluate_model,
    register_best_model,
    assert_minimum_auc,
)
from config import (
    get_data_paths,
    get_mlflow_config,
)

# Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Training config block (read directly to avoid import loop) ────────────────
import yaml as _yaml

_CONFIG_FILE = os.path.join(ROOT, "config", "config.yaml")


def _load_training_config() -> Dict:
    try:
        with open(_CONFIG_FILE, "r") as fh:
            cfg = _yaml.safe_load(fh)
        return cfg.get("training", {})
    except Exception as exc:
        logger.warning(f"Could not load training config: {exc}. Using defaults.")
        return {}


# Main pipeline function ─────────────────────────────────────────────────────

def training_pipeline(
    model_types: Optional[List[str]] = None,
    min_auc: float = 0.75,
    force_retrain: bool = False,
) -> Dict:
    """
    End-to-end training pipeline.

    1. Loads preprocessed CSVs produced by `data_pipeline.py`.
    2. Trains each model type in `model_types` under its own MLflow run.
    3. Selects the model with the highest test AUC-ROC.
    4. Checks the AUC quality gate (raises if AUC < min_auc).
    5. Registers the winner in the MLflow Model Registry.
    6. Returns a summary dict.

    Args:
        model_types:   List of model keys to train. Supported: 'logistic_regression', 'lightgbm'.
                       Defaults to both.
        min_auc:       Minimum AUC-ROC gate — pipeline raises ValueError if not met.
        force_retrain: If True, re-train even if artifacts/models/best_model_meta.json exists.

    Returns:
        Dict with keys: best_model_type, best_threshold, metrics, model_uri
    """
    if model_types is None:
        model_types = ["logistic_regression", "lightgbm"]

    logger.info("=" * 80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info(f"  Models : {model_types}")
    logger.info(f"  Min AUC: {min_auc}")
    logger.info("=" * 80)

    # Short-circuit if already trained ──────────────────────────────────────
    meta_path = "artifacts/models/best_model_meta.json"
    if not force_retrain and os.path.exists(meta_path):
        import json
        with open(meta_path) as fh:
            cached = json.load(fh)
        logger.info(f"  Trained model meta already exists at '{meta_path}'.")
        logger.info("  Pass force_retrain=True to re-train. Returning cached metadata.")
        return cached

    # Load data ─────────────────────────────────────────────────────────────
    data_paths = get_data_paths()
    X_train, X_test, y_train, y_test = load_processed_data(data_paths)

    # Use 20% of training set as validation for threshold optimisation
    val_size  = max(1, int(0.20 * len(X_train)))
    X_val     = X_train[-val_size:]
    y_val     = y_train[-val_size:]
    X_tr_only = X_train[:-val_size]
    y_tr_only = y_train[:-val_size]

    # MLflow setup ──────────────────────────────────────────────────────────
    mlflow_config = get_mlflow_config()
    tracking_uri  = mlflow_config.get("tracking_uri", "file:./mlruns")
    experiment_name = mlflow_config.get("experiment_name", "E-Commerce Fraud Detection")
    run_prefix    = mlflow_config.get("run_name_prefix", "fd_run")
    tags          = mlflow_config.get("tags", {})

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load training config ───────────────────────────────────────────────────
    training_config = _load_training_config()
    cost_matrix     = training_config.get("cost_matrix", {"fp_cost": 5.0, "fn_cost": 100.0})
    thresh_cfg      = training_config.get("threshold_range", {"start": 0.05, "stop": 0.96, "step": 0.01})
    threshold_range = (
        thresh_cfg.get("start", 0.05),
        thresh_cfg.get("stop",  0.96),
        thresh_cfg.get("step",  0.01),
    )

    results = {}

    # Train each model ──────────────────────────────────────────────────────
    for model_type in model_types:
        run_name = f"{run_prefix}_{model_type}"
        logger.info(f"\n{'─' * 70}")
        logger.info(f"  Starting MLflow run: {run_name}")
        logger.info(f"{'─' * 70}")

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("train_samples", len(X_tr_only))
            mlflow.log_param("test_samples",  len(X_test))
            mlflow.log_param("n_features",    X_train.shape[1])

            # Train ─────────────────────────────────────────────────────────
            if model_type == "logistic_regression":
                model = train_logistic_regression(
                    X_tr_only, y_tr_only, training_config, run
                )
            elif model_type == "lightgbm":
                model = train_lightgbm(
                    X_tr_only, y_tr_only, training_config, run
                )
            else:
                logger.warning(f"  Unknown model type '{model_type}' — skipping.")
                continue

            # Threshold optimisation ─────────────────────────────────────────
            threshold, expected_cost = optimize_threshold(
                model, X_val, y_val, cost_matrix, threshold_range
            )

            # Test evaluation ────────────────────────────────────────────────
            metrics = evaluate_model(
                model, X_test, y_test,
                threshold=threshold,
                model_name=model_type,
                cost_matrix=cost_matrix,
            )

            results[model_type] = {
                "model":     model,
                "threshold": threshold,
                "metrics":   metrics,
                "run_id":    run.info.run_id,
            }

        logger.info(f"  [{model_type}]  AUC={metrics['auc_roc']:.4f}  "
                    f"F1={metrics['f1_score']:.4f}  "
                    f"Threshold={threshold:.2f}")

    if not results:
        raise RuntimeError("No models were trained. Check model_types parameter.")

    # Select best model by test AUC ─────────────────────────────────────────
    best_type = max(results, key=lambda k: results[k]["metrics"]["auc_roc"])
    best      = results[best_type]
    logger.info(f"\n  Best model: {best_type}  (AUC={best['metrics']['auc_roc']:.4f})")

    # AUC quality gate ──────────────────────────────────────────────────────
    assert_minimum_auc(best["metrics"], min_auc=min_auc)

    # Register best model (inside a dedicated registration run) ─────────────
    reg_run_name = f"{run_prefix}_{best_type}_registration"
    with mlflow.start_run(run_name=reg_run_name, tags=tags) as reg_run:
        # Re-log key metrics so this run is self-contained
        for k, v in best["metrics"].items():
            mlflow.log_metric(k, v)
        mlflow.log_param("best_model_type", best_type)
        mlflow.log_param("optimal_threshold", best["threshold"])

        model_uri = register_best_model(
            model=best["model"],
            model_name=best_type,
            threshold=best["threshold"],
            metrics=best["metrics"],
            X_sample=X_train[:100],
            mlflow_config=mlflow_config,
        )

    # Summary ───────────────────────────────────────────────────────────────
    summary = {
        "best_model_type": best_type,
        "best_threshold":  best["threshold"],
        "metrics":         best["metrics"],
        "model_uri":       model_uri,
    }

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"  Best model    : {best_type}")
    logger.info(f"  AUC-ROC       : {best['metrics']['auc_roc']:.4f}")
    logger.info(f"  AUC-PR        : {best['metrics']['auc_pr']:.4f}")
    logger.info(f"  Precision     : {best['metrics']['precision']:.4f}")
    logger.info(f"  Recall        : {best['metrics']['recall']:.4f}")
    logger.info(f"  F1-Score      : {best['metrics']['f1_score']:.4f}")
    logger.info(f"  Threshold     : {best['threshold']:.2f}")
    logger.info(f"  Cost savings  : ${best['metrics']['cost_savings']:,.0f}")
    logger.info(f"  MLflow URI    : {model_uri}")
    logger.info("=" * 80)

    return summary


# CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Training Pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "lightgbm"],
        choices=["logistic_regression", "lightgbm"],
        help="Model types to train (default: both)",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.75,
        help="Minimum AUC-ROC gate threshold (default: 0.75)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Re-train even if model artifacts already exist",
    )
    args = parser.parse_args()

    result = training_pipeline(
        model_types=args.models,
        min_auc=args.min_auc,
        force_retrain=args.force_retrain,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model  : {result['best_model_type']}")
    print(f"  AUC-ROC     : {result['metrics']['auc_roc']:.4f}")
    print(f"  Threshold   : {result['best_threshold']:.2f}")
    print(f"  Cost savings: ${result['metrics']['cost_savings']:,.0f}")
    print(f"  MLflow URI  : {result['model_uri']}")
    print("=" * 60)
