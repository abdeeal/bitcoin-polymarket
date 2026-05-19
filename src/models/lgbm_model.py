"""
LightGBM Model — Primary ensemble member (weight: 0.50).

Gradient-boosted decision trees optimised for tabular data.
Handles high-cardinality features, missing values, and
feature interactions natively.

Data Leakage Prevention:
  - NEVER fits on validation/test data
  - Feature list frozen at training time — no post-hoc feature selection
  - Calibration is fitted on a HELD-OUT calibration set only
  - Hyperparameters tuned via walk-forward, not random holdout
"""

import os
import time
import json
from typing import Optional
from pathlib import Path

import numpy as np
import structlog

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logger = structlog.get_logger(__name__)

# ─── Default hyperparameters (PRD 6.3.1) ────────────────────────────
DEFAULT_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'num_leaves': 63,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


class LGBMModel:
    """
    LightGBM binary classifier wrapper.

    Provides a consistent interface for the ensemble:
      - fit(X, y, X_val, y_val) — train with early stopping
      - predict_proba(X)        — returns P(UP) for each sample
      - save(path) / load(path) — serialisation
      - feature_importance()    — for diagnostics
    """

    def __init__(self, params: Optional[dict] = None):
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._model = None
        self._feature_names: list[str] = []
        self._train_metadata: dict = {}

    # ────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        early_stopping_rounds: int = 50,
    ) -> dict:
        """
        Train on (X_train, y_train) with optional early stopping on validation.

        Data Leakage Guard:
          - X_val/y_val MUST be strictly chronologically after X_train
          - Caller is responsible for temporal ordering (WalkForwardValidator)

        Returns:
            dict with training metadata (best_iteration, train_loss, val_loss).
        """
        t0 = time.perf_counter()

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._model = lgb.LGBMClassifier(**self._params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['callbacks'] = [
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self._model.fit(X_train, y_train, **fit_kwargs)

        elapsed = time.perf_counter() - t0
        best_iter = getattr(self._model, 'best_iteration_', self._params['n_estimators'])

        self._train_metadata = {
            'model_type': 'lightgbm',
            'n_samples_train': len(y_train),
            'n_samples_val': len(y_val) if y_val is not None else 0,
            'n_features': X_train.shape[1],
            'best_iteration': best_iter,
            'train_time_sec': round(elapsed, 2),
            'params': self._params,
        }

        logger.info(
            "lgbm_trained",
            samples=len(y_train),
            features=X_train.shape[1],
            best_iter=best_iter,
            elapsed_sec=round(elapsed, 2),
        )
        return self._train_metadata

    # ────────────────────────────────────────────────────────────────────
    # Prediction
    # ────────────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(UP) for each sample. Shape: (n_samples,)."""
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")
        probs = self._model.predict_proba(X)
        return probs[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction (0/1) using 0.5 threshold."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ────────────────────────────────────────────────────────────────────
    # Feature Importance
    # ────────────────────────────────────────────────────────────────────

    def feature_importance(self, importance_type: str = 'gain') -> dict[str, float]:
        """Return feature importance dict sorted by value descending."""
        if self._model is None:
            return {}
        imp = self._model.feature_importances_
        pairs = sorted(
            zip(self._feature_names, imp),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: float(val) for name, val in pairs}

    # ────────────────────────────────────────────────────────────────────
    # Serialisation
    # ────────────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save model + metadata to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self._model:
            self._model.booster_.save_model(str(path / 'lgbm_model.txt'))
        with open(path / 'lgbm_metadata.json', 'w') as f:
            json.dump({
                'feature_names': self._feature_names,
                'train_metadata': self._train_metadata,
            }, f, indent=2)
        logger.info("lgbm_saved", path=str(path))

    @classmethod
    def load(cls, directory: str) -> 'LGBMModel':
        """Load a pre-trained model from directory."""
        path = Path(directory)
        instance = cls()
        booster = lgb.Booster(model_file=str(path / 'lgbm_model.txt'))
        instance._model = lgb.LGBMClassifier()
        instance._model._Booster = booster
        instance._model._n_classes = 2
        instance._model.fitted_ = True

        meta_path = path / 'lgbm_metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            instance._feature_names = meta.get('feature_names', [])
            instance._train_metadata = meta.get('train_metadata', {})

        logger.info("lgbm_loaded", path=str(path))
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
