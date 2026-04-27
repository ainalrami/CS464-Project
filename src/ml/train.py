"""
Classical ML model training with hyperparameter tuning.

Supports: SVM, Random Forest, XGBoost (fallback: Logistic Regression).
"""

import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Try to import XGBoost; fall back to Logistic Regression if unavailable
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost is available.")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost not installed. Using Logistic Regression as fallback.")


def _build_estimator(model_name, random_seed=42):
    """
    Return an sklearn estimator and its hyperparameter search space prefix.

    Args:
        model_name: One of "SVM", "RandomForest", "XGBoost".
        random_seed: Random seed passed to the estimator.

    Returns:
        (estimator, param_prefix) tuple.
    """
    if model_name == "SVM":
        return SVC(probability=True, random_state=random_seed), "clf__"
    elif model_name == "RandomForest":
        return RandomForestClassifier(random_state=random_seed, n_jobs=-1), "clf__"
    elif model_name == "XGBoost":
        if XGBOOST_AVAILABLE:
            return XGBClassifier(
                eval_metric="mlogloss",
                random_state=random_seed,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",   # histogram-based: 10-50x faster than exact
            ), "clf__"
        else:
            logger.warning("XGBoost not found — using Logistic Regression instead.")
            return LogisticRegression(
                max_iter=1000,
                random_state=random_seed,
                n_jobs=-1,
            ), "clf__"
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _get_actual_model_name(model_name):
    """Return the real model name accounting for XGBoost fallback."""
    if model_name == "XGBoost" and not XGBOOST_AVAILABLE:
        return "LogisticRegression"
    return model_name


def train_model(X_train, y_train, model_cfg, random_seed=42):
    """
    Train a single ML model with hyperparameter tuning.

    Args:
        X_train: Training feature matrix (N, D).
        y_train: Training labels (N,).
        model_cfg: Dict from config with keys: name, search, cv_folds, n_iter.
        random_seed: Random seed.

    Returns:
        best_pipeline: Fitted sklearn Pipeline (scaler + best classifier).
        best_params: Best hyperparameter dict.
        cv_score: Best cross-validation score.
        actual_name: The actual model name used (may differ if XGBoost falls back).
    """
    model_name = model_cfg["name"]
    actual_name = _get_actual_model_name(model_name)
    search_space = model_cfg.get("search", {})
    cv_folds = model_cfg.get("cv_folds", 3)
    n_iter = model_cfg.get("n_iter", 5)

    estimator, prefix = _build_estimator(model_name, random_seed)

    # Build pipeline: StandardScaler → Classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])

    # Prefix all search params with 'clf__'
    param_dist = {}
    for key, values in search_space.items():
        # Handle None values in YAML (loaded as Python None)
        param_dist[prefix + key] = values

    # If XGBoost fell back to LogReg, adapt the search space
    if model_name == "XGBoost" and not XGBOOST_AVAILABLE:
        param_dist = {
            prefix + "C": [0.01, 0.1, 1.0, 10.0],
            prefix + "solver": ["lbfgs", "saga"],
        }

    logger.info(f"Training {actual_name} with RandomizedSearchCV "
                f"(n_iter={n_iter}, cv={cv_folds})...")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=min(n_iter, _count_combinations(param_dist)),
        cv=cv_folds,
        scoring="accuracy",
        random_state=random_seed,
        n_jobs=-1,
        verbose=model_cfg.get("verbose", 2),
    )
    search.fit(X_train, y_train)

    logger.info(f"  Best CV accuracy: {search.best_score_:.4f}")
    logger.info(f"  Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_, actual_name


def _count_combinations(param_dist):
    """Count total combinations in parameter distribution."""
    count = 1
    for values in param_dist.values():
        if isinstance(values, list):
            count *= len(values)
    return count


def save_model(pipeline, model_name, output_dir):
    """
    Save a trained pipeline to disk.

    Args:
        pipeline: Fitted sklearn Pipeline.
        model_name: Name string for the filename.
        output_dir: Directory to save to.

    Returns:
        Path to saved model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}.pkl"
    joblib.dump(pipeline, path)
    logger.info(f"  Model saved to {path}")
    return path
