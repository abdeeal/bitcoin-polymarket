"""
Training Pipeline — End-to-end model training with leakage prevention.

Orchestrates:
  1. Data loading from TimescaleDB or CSV
  2. Label computation (with proper shift and NaN handling)
  3. Feature selection and ordering
  4. Walk-forward validation
  5. Final model training on latest data
  6. Model serialisation

Data Leakage Prevention Checklist:
  ☑ Labels computed via shift(-5) — future-looking by design (it's the target)
  ☑ Last 5 rows of each dataset have NaN labels — excluded from training
  ☑ Walk-forward enforces temporal ordering with purge gaps
  ☑ Normalisation stats from training set only (LSTM)
  ☑ Calibration on held-out set only (never train/val)
  ☑ No global feature engineering (all features are per-bar, no lookahead)
"""

import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from src.models.ensemble import EnsembleModel
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.models.lstm_model import LSTMModel
from src.models.calibration import PlattCalibrator
from src.training.walk_forward import WalkForwardValidator, LABEL_HORIZON

logger = structlog.get_logger(__name__)

# Feature columns to EXCLUDE from training (metadata, not predictive)
EXCLUDE_FEATURES = {
    'bot_state', 'timestamp', 'buffer_size', '_insufficient_data',
}


class TrainingPipeline:
    """
    End-to-end training pipeline with built-in leakage prevention.

    Usage:
        pipeline = TrainingPipeline()
        result = pipeline.train_from_features(features_matrix, save_dir='models/latest')
    """

    def __init__(
        self,
        walk_forward_config: Optional[dict] = None,
        model_save_dir: str = 'models/latest',
    ):
        self._wf_config = walk_forward_config or {}
        self._save_dir = model_save_dir
        self._feature_names: list[str] = []

    def compute_labels(
        self,
        close_prices: np.ndarray,
        horizon: int = LABEL_HORIZON,
    ) -> np.ndarray:
        """
        Compute binary labels: 1 if price goes UP in `horizon` bars, else 0.

        Label at time T = 1 if close[T+horizon] > close[T], else 0.
        Last `horizon` rows will be NaN (no future data available).

        Args:
            close_prices: 1D array of close prices, chronological.
            horizon: Number of bars to look ahead (default: 5 for 5-min bars = 25 min).

        Returns:
            1D array of labels with NaN at the end.
        """
        n = len(close_prices)
        labels = np.full(n, np.nan, dtype=np.float64)

        for i in range(n - horizon):
            labels[i] = 1.0 if close_prices[i + horizon] > close_prices[i] else 0.0

        logger.info(
            "labels_computed",
            n_samples=n,
            n_valid=int(np.sum(~np.isnan(labels))),
            n_nan=int(np.sum(np.isnan(labels))),
            class_balance=round(float(np.nanmean(labels)), 3),
        )
        return labels

    def prepare_features(
        self,
        feature_dicts: list[dict],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Convert list of feature dicts to 2D numpy array.

        Ensures consistent feature ordering and excludes metadata keys.

        Args:
            feature_dicts: List of dicts from FeatureEngine.compute_sync().

        Returns:
            (X, feature_names) — 2D array and ordered feature name list.
        """
        if not feature_dicts:
            return np.array([]).reshape(0, 0), []

        # Determine feature names from first dict (excluding metadata)
        all_keys = set()
        for fd in feature_dicts:
            all_keys.update(fd.keys())
        feature_names = sorted([k for k in all_keys if k not in EXCLUDE_FEATURES
                                and isinstance(feature_dicts[0].get(k, 0), (int, float))])

        self._feature_names = feature_names

        # Build matrix
        X = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float64)
        for i, fd in enumerate(feature_dicts):
            for j, name in enumerate(feature_names):
                val = fd.get(name, 0.0)
                X[i, j] = float(val) if isinstance(val, (int, float)) and not np.isnan(float(val)) else 0.0

        logger.info(
            "features_prepared",
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )
        return X, feature_names

    def train_from_features(
        self,
        X: np.ndarray,
        close_prices: np.ndarray,
        feature_names: list[str],
        run_walk_forward: bool = True,
        save: bool = True,
    ) -> dict:
        """
        Full training pipeline.

        Args:
            X: 2D feature matrix (n_samples, n_features), chronological.
            close_prices: 1D close prices aligned with X.
            feature_names: Feature name list.
            run_walk_forward: Whether to run walk-forward validation.
            save: Whether to save the trained model.

        Returns:
            dict with training results and walk-forward metrics.
        """
        t0 = time.perf_counter()
        self._feature_names = feature_names
        results = {}

        # 1. Compute labels
        y = self.compute_labels(close_prices)

        # 2. Run walk-forward validation (optional but recommended)
        if run_walk_forward:
            logger.info("pipeline_walk_forward_start")
            wf_validator = WalkForwardValidator(**self._wf_config)
            ensemble_wf = EnsembleModel()
            wf_result = wf_validator.run(X, y, ensemble_wf, feature_names)
            results['walk_forward'] = wf_result.summary()
            logger.info(
                "pipeline_walk_forward_complete",
                mean_accuracy=round(wf_result.mean_accuracy, 4),
                n_folds=len(wf_result.folds),
            )

        # 3. Train final model on latest available data
        logger.info("pipeline_final_training_start")
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Split: last 15% for val+cal, rest for training
        n = len(y_valid)
        train_end = int(n * 0.80)
        val_end = int(n * 0.90)

        X_train = X_valid[:train_end]
        y_train = y_valid[:train_end]
        X_val = X_valid[train_end:val_end]
        y_val = y_valid[train_end:val_end]
        X_cal = X_valid[val_end:]
        y_cal = y_valid[val_end:]

        ensemble = EnsembleModel()
        train_result = ensemble.fit_all(
            X_train, y_train,
            X_val, y_val,
            X_cal, y_cal,
            feature_names,
        )
        results['final_training'] = train_result

        # 4. Save model
        if save:
            ensemble.save(self._save_dir)
            # Also save feature names for inference
            with open(Path(self._save_dir) / 'feature_names.json', 'w') as f:
                json.dump(feature_names, f)
            results['save_path'] = self._save_dir

        elapsed = time.perf_counter() - t0
        results['total_pipeline_time_sec'] = round(elapsed, 1)

        logger.info("pipeline_complete", elapsed_sec=round(elapsed, 1))
        return results

    def train_from_local(
        self,
        data_dir: str = 'data/processed',
    ) -> dict:
        """
        Train from local .npy files.

        Fetches feature rows from X_180d.npy and close prices
        from close_180d.npy, and runs the full pipeline.
        """
        import os
        import json
        logger.info("loading_local_data", data_dir=data_dir)

        x_path = os.path.join(data_dir, 'X_180d.npy')
        close_path = os.path.join(data_dir, 'close_180d.npy')
        features_path = os.path.join(data_dir, 'feature_names.json')

        if not os.path.exists(x_path) or not os.path.exists(close_path) or not os.path.exists(features_path):
            raise FileNotFoundError(f"Missing data files in {data_dir}. Run download_and_extract.py first.")

        X = np.load(x_path)
        close_prices = np.load(close_path)

        with open(features_path, 'r') as f:
            feature_names = json.load(f)

        return self.train_from_features(X, close_prices, feature_names)

