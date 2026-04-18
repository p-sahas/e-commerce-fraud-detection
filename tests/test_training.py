"""
test_training.py — Unit tests for src/training.py

Covers:
  - assert_minimum_auc: gate passing and failing
  - optimize_threshold: returns (float, float), cost decreases vs 0.5 default
  - evaluate_model:     all metric keys present, values in valid ranges
  - Cost matrix arithmetic: FP/FN cost calculation correctness
  - Model training functions: train_logistic_regression / train_lightgbm
    (smoke tests — verify they fit, predict_proba works, AUC > 0.5)
"""

import os
import sys
import types

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))


# =============================================================================
# Helpers
# =============================================================================

def _make_imbalanced_dataset(n=400, fraud_rate=0.10, seed=0):
    """Create imbalanced binary classification data with structure (auc > 0.5)."""
    X, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        weights=[1 - fraud_rate, fraud_rate],
        random_state=seed,
        flip_y=0.02,
    )
    return X.astype(np.float32), y


def _fit_lr(X, y):
    """Quick logistic regression fit for test isolation."""
    model = LogisticRegression(max_iter=200, class_weight="balanced", random_state=0)
    model.fit(X, y)
    return model


# =============================================================================
# assert_minimum_auc
# =============================================================================

class TestAssertMinimumAuc:

    def test_passes_when_auc_above_gate(self):
        from training import assert_minimum_auc
        assert_minimum_auc({"auc_roc": 0.82}, min_auc=0.75)   # no exception

    def test_passes_exactly_at_gate(self):
        from training import assert_minimum_auc
        assert_minimum_auc({"auc_roc": 0.75}, min_auc=0.75)

    def test_raises_when_auc_below_gate(self):
        from training import assert_minimum_auc
        with pytest.raises(ValueError, match="below the minimum"):
            assert_minimum_auc({"auc_roc": 0.60}, min_auc=0.75)

    def test_raises_when_auc_missing(self):
        from training import assert_minimum_auc
        with pytest.raises(ValueError):
            assert_minimum_auc({}, min_auc=0.75)

    def test_default_gate_is_0_75(self):
        from training import assert_minimum_auc
        with pytest.raises(ValueError):
            assert_minimum_auc({"auc_roc": 0.74})

    def test_custom_gate_value(self):
        from training import assert_minimum_auc
        assert_minimum_auc({"auc_roc": 0.55}, min_auc=0.50)

    def test_raises_with_zero_auc(self):
        from training import assert_minimum_auc
        with pytest.raises(ValueError):
            assert_minimum_auc({"auc_roc": 0.0})


# =============================================================================
# optimize_threshold
# =============================================================================

class TestOptimizeThreshold:

    @pytest.fixture(autouse=True)
    def _mlflow_noop(self, monkeypatch):
        """Patch mlflow calls so threshold tests don't need a live tracking server."""
        import mlflow
        monkeypatch.setattr(mlflow, "log_metric", lambda *a, **kw: None)
        monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **kw: None)

    def test_returns_tuple_of_two_floats(self, sample_X_y, cost_matrix):
        from training import optimize_threshold
        X, y = sample_X_y
        model = _fit_lr(X, y)
        result = optimize_threshold(model, X, y, cost_matrix)
        assert isinstance(result, tuple) and len(result) == 2
        thresh, cost = result
        assert isinstance(thresh, float)
        assert isinstance(cost, float)

    def test_threshold_in_range(self, sample_X_y, cost_matrix):
        from training import optimize_threshold
        X, y = sample_X_y
        model = _fit_lr(X, y)
        thresh, _ = optimize_threshold(model, X, y, cost_matrix)
        assert 0.0 < thresh < 1.0

    def test_cost_is_non_negative(self, sample_X_y, cost_matrix):
        from training import optimize_threshold
        X, y = sample_X_y
        model = _fit_lr(X, y)
        _, cost = optimize_threshold(model, X, y, cost_matrix)
        assert cost >= 0.0

    def test_optimal_cost_not_worse_than_midpoint(self, cost_matrix):
        """Optimal threshold should produce <= cost at threshold=0.5."""
        from training import optimize_threshold
        from sklearn.metrics import confusion_matrix as cm

        X, y = _make_imbalanced_dataset(400)
        model = _fit_lr(X, y)

        thresh_opt, cost_opt = optimize_threshold(model, X, y, cost_matrix)

        # Cost at 0.5
        proba = model.predict_proba(X)[:, 1]
        preds_05 = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = cm(y, preds_05, labels=[0, 1]).ravel()
        cost_05 = fp * cost_matrix["fp_cost"] + fn * cost_matrix["fn_cost"]

        assert cost_opt <= cost_05, (
            f"Optimal cost ${cost_opt:,.0f} > midpoint cost ${cost_05:,.0f}"
        )

    def test_higher_fn_cost_lowers_threshold(self, sample_X_y):
        """When fraud losses are very high, the model should flag more aggressively."""
        from training import optimize_threshold
        X, y = sample_X_y
        model = _fit_lr(X, y)

        thresh_conservative, _ = optimize_threshold(model, X, y, {"fp_cost": 100, "fn_cost": 1})
        thresh_aggressive, _   = optimize_threshold(model, X, y, {"fp_cost": 1,   "fn_cost": 100})

        assert thresh_aggressive <= thresh_conservative, (
            "Higher FN cost should push threshold downward (flag more transactions)"
        )


# =============================================================================
# evaluate_model
# =============================================================================

class TestEvaluateModel:

    @pytest.fixture(autouse=True)
    def _mlflow_noop(self, monkeypatch):
        import mlflow
        monkeypatch.setattr(mlflow, "log_metric",   lambda *a, **kw: None)
        monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **kw: None)

    def _run_eval(self, sample_X_y, cost_matrix):
        from training import evaluate_model
        X, y = sample_X_y
        model = _fit_lr(X, y)
        return evaluate_model(model, X, y, threshold=0.5,
                              model_name="logistic_regression",
                              cost_matrix=cost_matrix)

    def test_returns_dict(self, sample_X_y, cost_matrix):
        metrics = self._run_eval(sample_X_y, cost_matrix)
        assert isinstance(metrics, dict)

    def test_all_required_keys_present(self, sample_X_y, cost_matrix):
        required = {
            "auc_roc", "auc_pr", "precision", "recall", "f1_score",
            "threshold", "tp", "fp", "tn", "fn",
            "expected_cost", "baseline_cost", "cost_savings",
        }
        metrics = self._run_eval(sample_X_y, cost_matrix)
        missing = required - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"

    def test_auc_in_valid_range(self, sample_X_y, cost_matrix):
        metrics = self._run_eval(sample_X_y, cost_matrix)
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert 0.0 <= metrics["auc_pr"]  <= 1.0

    def test_auc_roc_above_random_baseline(self, sample_X_y, cost_matrix):
        """A logistic regression on structured data must beat random (auc > 0.5)."""
        metrics = self._run_eval(sample_X_y, cost_matrix)
        assert metrics["auc_roc"] > 0.5, (
            f"AUC {metrics['auc_roc']:.4f} is not better than random chance"
        )

    def test_precision_recall_in_range(self, sample_X_y, cost_matrix):
        metrics = self._run_eval(sample_X_y, cost_matrix)
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"]    <= 1.0

    def test_confusion_matrix_counts_non_negative(self, sample_X_y, cost_matrix):
        metrics = self._run_eval(sample_X_y, cost_matrix)
        assert metrics["tp"] >= 0
        assert metrics["fp"] >= 0
        assert metrics["tn"] >= 0
        assert metrics["fn"] >= 0

    def test_cost_savings_arithmetic(self, sample_X_y, cost_matrix):
        metrics = self._run_eval(sample_X_y, cost_matrix)
        expected = round(metrics["baseline_cost"] - metrics["expected_cost"], 2)
        assert abs(metrics["cost_savings"] - expected) < 0.01

    def test_expected_cost_formula(self, sample_X_y, cost_matrix):
        metrics = self._run_eval(sample_X_y, cost_matrix)
        manual = (
            metrics["fp"] * cost_matrix["fp_cost"] +
            metrics["fn"] * cost_matrix["fn_cost"]
        )
        assert abs(metrics["expected_cost"] - manual) < 0.1


# =============================================================================
# Model training smoke tests
# =============================================================================

class TestTrainLogisticRegression:

    @pytest.fixture(autouse=True)
    def _mlflow_active_run(self, tmp_mlflow_dir):
        """Ensure an active MLflow run for functions that call mlflow.log_*."""
        import mlflow
        with mlflow.start_run():
            yield

    def test_returns_fitted_model(self, sample_X_y):
        from training import train_logistic_regression
        X, y = sample_X_y
        # train_logistic_regression expects pandas DataFrames and column names
        X_df = {"X_train": X, "y_train": y}
        model, metrics = train_logistic_regression(
            X_train=X,
            y_train=y,
            cv_splits=2,
        )
        assert hasattr(model, "predict_proba")

    def test_model_predict_proba_shape(self, sample_X_y):
        from training import train_logistic_regression
        X, y = sample_X_y
        model, _ = train_logistic_regression(X_train=X, y_train=y, cv_splits=2)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_returns_metrics_dict(self, sample_X_y):
        from training import train_logistic_regression
        X, y = sample_X_y
        _, metrics = train_logistic_regression(X_train=X, y_train=y, cv_splits=2)
        assert isinstance(metrics, dict)
        assert "auc_roc" in metrics


class TestTrainLightGBM:

    @pytest.fixture(autouse=True)
    def _mlflow_active_run(self, tmp_mlflow_dir):
        import mlflow
        with mlflow.start_run():
            yield

    def test_returns_fitted_model(self, sample_X_y):
        from training import train_lightgbm
        X, y = sample_X_y
        model, _ = train_lightgbm(X_train=X, y_train=y, cv_splits=2)
        assert hasattr(model, "predict_proba") or hasattr(model, "predict")

    def test_predict_proba_works(self, sample_X_y):
        from training import train_lightgbm
        X, y = sample_X_y
        model, _ = train_lightgbm(X_train=X, y_train=y, cv_splits=2)
        proba = model.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_auc_above_random(self, sample_X_y, cost_matrix):
        from training import train_lightgbm, evaluate_model
        X, y = sample_X_y
        model, _ = train_lightgbm(X_train=X, y_train=y, cv_splits=2)
        metrics = evaluate_model(model, X, y, 0.5, "lightgbm", cost_matrix)
        assert metrics["auc_roc"] > 0.5
