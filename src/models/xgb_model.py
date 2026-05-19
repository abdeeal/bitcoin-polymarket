"""
XGBoost Model — Diversity ensemble member (weight: 0.15).

Gradient-boosted trees with histogram-based split finding.
Provides model diversity via different boosting algorithm and
regularisation characteristics compared to LightGBM.

Data Leakage Prevention:
  - Same temporal split contract as LGBMModel
  - tree_method='hist' for deterministic GPU-free training
"""

import time
import json
from typing import Optional
from pathlib import Path

import numpy as np
import structlog

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = structlog.get_logger(__name__)

DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 800,
    'learning_rate': 0.025,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}


class XGBModel:
    """XGBoost binary classifier wrapper with consistent ensemble interface."""

    def __init__(self, params: Optional[dict] = None):
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._model = None
        self._feature_names: list[str] = []
        self._train_metadata: dict = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        early_stopping_rounds: int = 50,
    ) -> dict:
        t0 = time.perf_counter()
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._model = xgb.XGBClassifier(**self._params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['verbose'] = False

        self._model.fit(X_train, y_train, **fit_kwargs)

        elapsed = time.perf_counter() - t0
        best_iter = getattr(self._model, 'best_iteration', self._params['n_estimators'])

        self._train_metadata = {
            'model_type': 'xgboost',
            'n_samples_train': len(y_train),
            'n_samples_val': len(y_val) if y_val is not None else 0,
            'n_features': X_train.shape[1],
            'best_iteration': best_iter,
            'train_time_sec': round(elapsed, 2),
        }
        logger.info("xgb_trained", samples=len(y_train), elapsed_sec=round(elapsed, 2))
        return self._train_metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def feature_importance(self, importance_type: str = 'gain') -> dict[str, float]:
        if self._model is None:
            return {}
        imp = self._model.feature_importances_
        pairs = sorted(zip(self._feature_names, imp), key=lambda x: x[1], reverse=True)
        return {name: float(val) for name, val in pairs}

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self._model:
            self._model.save_model(str(path / 'xgb_model.json'))
        with open(path / 'xgb_metadata.json', 'w') as f:
            json.dump({'feature_names': self._feature_names, 'train_metadata': self._train_metadata}, f, indent=2)
        logger.info("xgb_saved", path=str(path))

    @classmethod
    def load(cls, directory: str) -> 'XGBModel':
        path = Path(directory)
        instance = cls()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(str(path / 'xgb_model.json'))
        meta_path = path / 'xgb_metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            instance._feature_names = meta.get('feature_names', [])
            instance._train_metadata = meta.get('train_metadata', {})
        logger.info("xgb_loaded", path=str(path))
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
