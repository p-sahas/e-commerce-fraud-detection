"""
Model training module for e-commerce fraud detection.

Provides:
  - load_processed_data     : Load CSV artifacts produced by the data pipeline
  - train_logistic_regression : Baseline model with TimeSeriesSplit grid search
  - train_lightgbm          : Advanced model with class-imbalance handling
  - optimize_threshold      : Cost-aware decision threshold sweep
  - evaluate_model          : Full evaluation suite logged to MLflow
  - register_best_model     : MLflow model registration + local metadata export
"""

import os
import sys
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)



# Data Loading


def load_processed_data(data_paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed CSV artifacts produced by the data pipeline into
    numpy arrays ready for scikit-learn / LightGBM training.

    Args:
        data_paths: Dict with keys 'X_train', 'X_test', 'Y_train', 'Y_test'
                    mapping to CSV file paths.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy float arrays.
    """
    logger.info("=" * 60)
    logger.info("LOADING PREPROCESSED DATA")
    logger.info("=" * 60)

    def _load(path: str, label: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{label}] CSV not found: {path}")
        df = pd.read_csv(path)
        logger.info(f"  {label}: {df.shape}  ->  {path}")
        return df

    X_train_df = _load(data_paths["X_train"], "X_train")
    X_test_df  = _load(data_paths["X_test"],  "X_test")
    y_train_df = _load(data_paths["Y_train"], "Y_train")
    y_test_df  = _load(data_paths["Y_test"],  "Y_test")

    # Flatten label columns to 1-D arrays
    y_train = y_train_df.iloc[:, 0].values.astype(int)
    y_test  = y_test_df.iloc[:, 0].values.astype(int)

    # Numeric cast — fill any remaining NaN with 0
    X_train = X_train_df.select_dtypes(include=[np.number]).fillna(0).values.astype(float)
    X_test  = X_test_df.select_dtypes(include=[np.number]).fillna(0).values.astype(float)

    logger.info(f"  X_train shape: {X_train.shape}  |  fraud rate: {y_train.mean():.4f}")
    logger.info(f"  X_test  shape: {X_test.shape}   |  fraud rate: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test



# Baseline — Logistic Regression


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    training_config: Dict[str, Any],
    mlflow_run: Any,
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model with TimeSeriesSplit
    cross-validation over a C-grid.  Best estimator is returned and
    its parameters + CV AUC are logged to the active MLflow run.

    Args:
        X_train:         Feature matrix (n_samples, n_features)
        y_train:         Target labels
        training_config: `training.logistic_regression` config block
        mlflow_run:      Active MLflow run object (for context only; auto-logging handles it)

    Returns:
        Fitted best LogisticRegression estimator
    """
    logger.info("=" * 60)
    logger.info("TRAINING — LOGISTIC REGRESSION (BASELINE)")
    logger.info("=" * 60)

    lr_cfg = training_config.get("logistic_regression", {})
    n_splits   = training_config.get("time_series_split", {}).get("n_splits", 5)
    C_grid     = lr_cfg.get("C", [0.01, 0.1, 1.0, 10.0])
    max_iter   = lr_cfg.get("max_iter", 1000)
    solver     = lr_cfg.get("solver", "lbfgs")
    class_wt   = lr_cfg.get("class_weight", "balanced")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    param_grid = {"C": C_grid}
    base_lr = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight=class_wt,
        random_state=42,
        n_jobs=-1,
    )

    gs = GridSearchCV(
        base_lr,
        param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(X_train, y_train)

    best_lr   = gs.best_estimator_
    best_auc  = gs.best_score_

    logger.info(f"  Best C   : {gs.best_params_['C']}")
    logger.info(f"  CV AUC   : {best_auc:.4f}")

    # Log to MLflow
    mlflow.log_param("lr_best_C",      gs.best_params_["C"])
    mlflow.log_param("lr_max_iter",    max_iter)
    mlflow.log_param("lr_solver",      solver)
    mlflow.log_param("lr_class_weight", class_wt)
    mlflow.log_param("lr_n_cv_splits",  n_splits)
    mlflow.log_metric("lr_cv_auc",     best_auc)

    # Per-fold CV results
    for fold_i, (auc_score) in enumerate(gs.cv_results_["mean_test_score"]):
        logger.info(f"  Param C={gs.cv_results_['param_C'][fold_i]} | mean AUC={auc_score:.4f}")

    return best_lr



# Advanced — LightGBM


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    training_config: Dict[str, Any],
    mlflow_run: Any,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier with class-imbalance handling via
    `scale_pos_weight`, using TimeSeriesSplit cross-validation with
    early stopping on the last fold.  Feature importances are saved
    as an MLflow artifact.

    Args:
        X_train:         Feature matrix (n_samples, n_features)
        y_train:         Target labels
        training_config: `training.lightgbm` config block
        mlflow_run:      Active MLflow run object

    Returns:
        Fitted LGBMClassifier
    """
    logger.info("=" * 60)
    logger.info("TRAINING — LIGHTGBM (ADVANCED)")
    logger.info("=" * 60)

    lgbm_cfg   = training_config.get("lightgbm", {})
    n_splits   = training_config.get("time_series_split", {}).get("n_splits", 5)

    # Class imbalance — ratio of negatives to positives
    neg   = int((y_train == 0).sum())
    pos   = int((y_train == 1).sum())
    ratio = round(neg / max(pos, 1), 2)
    logger.info(f"  class distribution  neg={neg}  pos={pos}  scale_pos_weight={ratio}")

    params = {
        "n_estimators":    lgbm_cfg.get("n_estimators",     300),
        "max_depth":       lgbm_cfg.get("max_depth",         6),
        "learning_rate":   lgbm_cfg.get("learning_rate",     0.05),
        "num_leaves":      lgbm_cfg.get("num_leaves",        63),
        "min_child_samples": lgbm_cfg.get("min_child_samples", 20),
        "subsample":       lgbm_cfg.get("subsample",         0.8),
        "colsample_bytree": lgbm_cfg.get("colsample_bytree", 0.8),
        "reg_alpha":       lgbm_cfg.get("reg_alpha",         0.1),
        "reg_lambda":      lgbm_cfg.get("reg_lambda",        0.1),
        "scale_pos_weight": lgbm_cfg.get("scale_pos_weight", ratio),
        "objective":       "binary",
        "metric":          "auc",
        "random_state":    42,
        "n_jobs":          -1,
        "verbose":         -1,
    }

    tscv      = TimeSeriesSplit(n_splits=n_splits)
    fold_aucs = []

    # Cross-validation for metric estimation
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        fold_auc  = roc_auc_score(y_val, val_proba)
        fold_aucs.append(fold_auc)
        logger.info(f"  Fold {fold + 1}/{n_splits}  |  val AUC={fold_auc:.4f}")

    mean_cv_auc = float(np.mean(fold_aucs))
    logger.info(f"  Mean CV AUC: {mean_cv_auc:.4f}")

    # Final fit on all training data
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_train, y_train)

    # Log to MLflow
    for k, v in params.items():
        try:
            mlflow.log_param(f"lgbm_{k}", v)
        except Exception:
            pass
    mlflow.log_metric("lgbm_cv_auc_mean", mean_cv_auc)
    for i, auc_ in enumerate(fold_aucs):
        mlflow.log_metric(f"lgbm_cv_auc_fold_{i + 1}", auc_)

    # Feature importance plot
    _log_feature_importance(final_model, X_train.shape[1])

    return final_model


def _log_feature_importance(model: lgb.LGBMClassifier, n_features: int) -> None:
    """Generate and log a feature importance bar chart to MLflow."""
    try:
        importances = model.feature_importances_
        feat_labels = [f"feat_{i}" for i in range(n_features)]

        top_n = min(20, n_features)
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [feat_labels[i] for i in indices][::-1],
            importances[indices][::-1],
            color="#4CAF50",
        )
        ax.set_xlabel("Importance (split)")
        ax.set_title("LightGBM — Top Feature Importances")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()

        os.makedirs("artifacts/plots", exist_ok=True)
        fig_path = "artifacts/plots/lgbm_feature_importance.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="plots")
        logger.info(f"  Feature importance plot saved → {fig_path}")
    except Exception as exc:
        logger.warning(f"  Could not generate feature importance plot: {exc}")



# Threshold Optimisation


def optimize_threshold(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cost_matrix: Dict[str, float],
    threshold_range: Tuple[float, float, float] = (0.05, 0.96, 0.01),
) -> Tuple[float, float]:
    """
    Sweep decision thresholds and pick the one that minimises expected cost.

    Expected cost = FP_count * C_fp  +  FN_count * C_fn

    Args:
        model:           Fitted classifier with predict_proba
        X_val:           Validation feature matrix
        y_val:           Validation labels
        cost_matrix:     Dict with 'fp_cost' and 'fn_cost' keys
        threshold_range: (start, stop, step) for np.arange sweep

    Returns:
        Tuple of (optimal_threshold, minimum_expected_cost)
    """
    logger.info("=" * 60)
    logger.info("THRESHOLD OPTIMISATION")
    logger.info("=" * 60)

    fp_cost = cost_matrix.get("fp_cost", 5.0)
    fn_cost = cost_matrix.get("fn_cost", 100.0)

    proba    = model.predict_proba(X_val)[:, 1]
    start, stop, step = threshold_range
    thresholds        = np.arange(start, stop, step)

    best_thresh = 0.5
    best_cost   = float("inf")
    costs       = []

    for thresh in thresholds:
        preds = (proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

        expected_cost = fp * fp_cost + fn * fn_cost
        costs.append(expected_cost)

        if expected_cost < best_cost:
            best_cost   = expected_cost
            best_thresh = float(thresh)

    logger.info(f"  FP cost=${fp_cost}  FN cost=${fn_cost}")
    logger.info(f"  Optimal threshold : {best_thresh:.2f}")
    logger.info(f"  Minimum expected cost : ${best_cost:,.0f}")

    # Log threshold curve
    _log_threshold_curve(thresholds, costs, best_thresh)

    mlflow.log_metric("optimal_threshold",    best_thresh)
    mlflow.log_metric("min_expected_cost",    best_cost)

    return best_thresh, best_cost


def _log_threshold_curve(thresholds: np.ndarray, costs: List[float], best_thresh: float) -> None:
    """Save threshold vs expected-cost plot as MLflow artifact."""
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, costs, color="#2196F3", linewidth=2)
        ax.axvline(best_thresh, color="#F44336", linestyle="--", label=f"Optimal={best_thresh:.2f}")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Expected Cost ($)")
        ax.set_title("Threshold vs. Expected Business Cost")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()

        os.makedirs("artifacts/plots", exist_ok=True)
        fig_path = "artifacts/plots/threshold_cost_curve.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="plots")
    except Exception as exc:
        logger.warning(f"  Could not save threshold curve: {exc}")



# Evaluation


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    model_name: str,
    cost_matrix: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute and log a comprehensive set of evaluation metrics.

    Metrics logged:
      - AUC-ROC, AUC-PR
      - Precision, Recall, F1 at the given threshold
      - Confusion matrix counts (TP, FP, TN, FN)
      - Expected cost and expected cost savings vs. no-model baseline

    Args:
        model:       Fitted classifier
        X_test:      Test feature matrix
        y_test:      Test labels
        threshold:   Decision threshold from optimize_threshold
        model_name:  Short name string (e.g. "logistic_regression")
        cost_matrix: Dict with 'fp_cost' and 'fn_cost' keys

    Returns:
        Dict of metric name → value
    """
    logger.info("=" * 60)
    logger.info(f"EVALUATION — {model_name.upper()}")
    logger.info("=" * 60)

    fp_cost = cost_matrix.get("fp_cost",  5.0)
    fn_cost = cost_matrix.get("fn_cost", 100.0)

    proba   = model.predict_proba(X_test)[:, 1]
    preds   = (proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, proba)
    auc_pr  = average_precision_score(y_test, proba)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)
    f1      = f1_score(y_test, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    expected_cost  = fp * fp_cost + fn * fn_cost

    # Baseline: flag every transaction → zero FN but max FP
    baseline_cost  = int((y_test == 0).sum()) * fp_cost
    cost_savings   = baseline_cost - expected_cost

    metrics = {
        "auc_roc":          round(auc_roc, 4),
        "auc_pr":           round(auc_pr, 4),
        "precision":        round(prec, 4),
        "recall":           round(rec, 4),
        "f1_score":         round(f1, 4),
        "threshold":        round(threshold, 4),
        "tp":               int(tp),
        "fp":               int(fp),
        "tn":               int(tn),
        "fn":               int(fn),
        "expected_cost":    round(expected_cost, 2),
        "baseline_cost":    round(baseline_cost, 2),
        "cost_savings":     round(cost_savings, 2),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    logger.info(f"  AUC-ROC  : {auc_roc:.4f}")
    logger.info(f"  AUC-PR   : {auc_pr:.4f}")
    logger.info(f"  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
    logger.info(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    logger.info(f"  Expected cost  : ${expected_cost:,.0f}")
    logger.info(f"  Cost savings   : ${cost_savings:,.0f} (vs flag-all baseline)")

    # Plots
    _log_roc_curve(model, X_test, y_test, model_name)
    _log_pr_curve(model, X_test, y_test, model_name)
    _log_confusion_matrix_plot(tn, fp, fn, tp, model_name)

    return metrics


def _log_roc_curve(model: Any, X_test: np.ndarray, y_test: np.ndarray, tag: str) -> None:
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=tag)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title(f"ROC Curve — {tag}")
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        os.makedirs("artifacts/plots", exist_ok=True)
        path = f"artifacts/plots/roc_curve_{tag}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="plots")
    except Exception as exc:
        logger.warning(f"  ROC curve error: {exc}")


def _log_pr_curve(model: Any, X_test: np.ndarray, y_test: np.ndarray, tag: str) -> None:
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=tag)
        ax.set_title(f"Precision-Recall Curve — {tag}")
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        os.makedirs("artifacts/plots", exist_ok=True)
        path = f"artifacts/plots/pr_curve_{tag}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="plots")
    except Exception as exc:
        logger.warning(f"  PR curve error: {exc}")


def _log_confusion_matrix_plot(tn: int, fp: int, fn: int, tp: int, tag: str) -> None:
    try:
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                        fontsize=13, color="black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
        ax.set_yticklabels(["Actual Negative", "Actual Positive"])
        ax.set_title(f"Confusion Matrix — {tag}")
        plt.tight_layout()
        os.makedirs("artifacts/plots", exist_ok=True)
        path = f"artifacts/plots/confusion_matrix_{tag}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="plots")
    except Exception as exc:
        logger.warning(f"  Confusion matrix plot error: {exc}")



# Model Registration


def register_best_model(
    model: Any,
    model_name: str,
    threshold: float,
    metrics: Dict[str, float],
    X_sample: np.ndarray,
    mlflow_config: Dict[str, Any],
) -> str:
    """
    Register the winning model in the MLflow Model Registry and export
    a local JSON metadata file to `artifacts/models/best_model_meta.json`.

    Args:
        model:         Fitted best model (sklearn or lgbm compat)
        model_name:    'logistic_regression' or 'lightgbm'
        threshold:     Optimal decision threshold
        metrics:       Dict of evaluation metrics
        X_sample:      Small slice of X_train for signature inference
        mlflow_config: `mlflow` section of config.yaml

    Returns:
        MLflow model URI string
    """
    logger.info("=" * 60)
    logger.info("REGISTERING BEST MODEL")
    logger.info("=" * 60)

    registry_name = mlflow_config.get("model_registry_name", "fraud_detection")
    artifact_path = mlflow_config.get("artifact_path", "model")

    # Infer signature from a sample
    sample_preds = model.predict_proba(X_sample[:5])
    signature    = infer_signature(X_sample[:5], sample_preds)

    # Log model to current run's artifact store
    if model_name == "lightgbm":
        mlflow.lightgbm.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=registry_name,
        )
    else:
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=registry_name,
        )

    run_id  = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"  Registered '{registry_name}'  →  {model_uri}")

    # Local metadata export
    os.makedirs("artifacts/models", exist_ok=True)
    meta = {
        "model_type":  model_name,
        "threshold":   threshold,
        "registry":    registry_name,
        "mlflow_uri":  model_uri,
        "run_id":      run_id,
        "metrics":     metrics,
    }
    meta_path = "artifacts/models/best_model_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info(f"  Metadata saved → {meta_path}")

    mlflow.log_artifact(meta_path, artifact_path="model_meta")

    return model_uri



# AUC Gate


def assert_minimum_auc(metrics: Dict[str, float], min_auc: float = 0.75) -> None:
    """
    Raise ValueError if test AUC-ROC falls below the minimum threshold.
    This acts as the CI quality gate for model performance.

    Args:
        metrics:  Evaluation metrics dict containing 'auc_roc'
        min_auc:  Minimum acceptable AUC (default 0.75)
    """
    auc = metrics.get("auc_roc", 0.0)
    if auc < min_auc:
        raise ValueError(
            f"Model AUC {auc:.4f} is below the minimum required AUC of {min_auc:.2f}. "
            "Pipeline halted — check features, data quality, or model config."
        )
    logger.info(f"  AUC gate passed: {auc:.4f} >= {min_auc:.2f}")
