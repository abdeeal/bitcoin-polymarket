"""
Ensemble Model — Weighted combination of LightGBM + LSTM + XGBoost.

Architecture (PRD 6.3.4):
  P(UP) = 0.50 × P_lgbm + 0.35 × P_lstm + 0.15 × P_xgb

Calibration:
  - Raw ensemble output is passed through PlattCalibrator
  - Calibration fitted on a HELD-OUT calibration set (never training data)
  - Target Expected Calibration Error (ECE) < 0.03

Data Leakage Prevention:
  - Ensemble weights are fixed (not learned from data)
  - Calibration uses a separate temporal split (after val, before test)
  - Each sub-model trained on identical temporal splits
  - Model versioning: each save includes training window metadata
"""

import time
import json
from typing import Optional
from pathlib import Path

import numpy as np
import structlog

from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.models.lstm_model import LSTMModel
from src.models.calibration import PlattCalibrator

logger = structlog.get_logger(__name__)

# PRD-defined ensemble weights
WEIGHTS = {
    'lgbm': 0.50,
    'lstm': 0.35,
    'xgb': 0.15,
}


class EnsembleModel:
    """
    Weighted ensemble of LightGBM + LSTM + XGBoost.

    Inference flow:
      1. Each sub-model produces P(UP) independently
      2. Weighted average produces raw ensemble probability
      3. PlattCalibrator maps raw prob to calibrated probability
      4. Calibrated probability is returned to the signal generator

    The ensemble NEVER learns weights from data — they are fixed
    per PRD to avoid overfitting on the meta-learner level.
    """

    def __init__(
        self,
        lgbm: Optional[LGBMModel] = None,
        xgb_model: Optional[XGBModel] = None,
        lstm: Optional[LSTMModel] = None,
        calibrator: Optional[PlattCalibrator] = None,
        weights: Optional[dict] = None,
    ):
        self.lgbm = lgbm or LGBMModel()
        self.xgb = xgb_model or XGBModel()
        self.lstm = lstm or LSTMModel()
        self.calibrator = calibrator or PlattCalibrator()
        self.weights = weights or WEIGHTS.copy()
        self._metadata: dict = {}

    # ────────────────────────────────────────────────────────────────────
    # Prediction
    # ────────────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        X_flat: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
        calibrate: bool = True,
    ) -> np.ndarray:
        """
        Produce calibrated P(UP) from the ensemble.

        Args:
            X_flat: 2D array (n_samples, n_features) for tree models.
            X_seq:  3D array (n_samples, seq_len, n_features) for LSTM.
                    If None, LSTM contribution is redistributed.
            calibrate: Whether to apply Platt calibration.

        Returns:
            1D array of P(UP) values, shape (n_samples,).
        """
        probs = {}
        active_weight = 0.0

        # LightGBM
        if self.lgbm.is_fitted:
            probs['lgbm'] = self.lgbm.predict_proba(X_flat)
            active_weight += self.weights['lgbm']

        # XGBoost
        if self.xgb.is_fitted:
            probs['xgb'] = self.xgb.predict_proba(X_flat)
            active_weight += self.weights['xgb']

        # LSTM
        if self.lstm.is_fitted and X_seq is not None:
            probs['lstm'] = self.lstm.predict_proba(X_seq)
            active_weight += self.weights['lstm']

        if not probs:
            return np.full(len(X_flat), 0.5)

        # Weighted combination (re-normalise if not all models active)
        result = np.zeros(len(X_flat))
        for name, prob in probs.items():
            w = self.weights[name] / active_weight
            result += w * prob

        # Calibration
        if calibrate and self.calibrator.is_fitted:
            result = self.calibrator.transform(result)

        return result

    def predict_single(
        self,
        features: dict,
        feature_names: list[str],
        X_seq: Optional[np.ndarray] = None,
    ) -> float:
        """
        Predict P(UP) for a single bar.

        Convenience method for live inference — constructs arrays
        from a feature dict.

        Args:
            features: dict of feature_name -> scalar value.
            feature_names: ordered list of feature names for the model.
            X_seq: Optional 3D sequence array for LSTM.

        Returns:
            Calibrated P(UP) as a scalar float.
        """
        X = np.array([[features.get(f, 0.0) for f in feature_names]])
        probs = self.predict_proba(X, X_seq)
        return float(probs[0])

    def predict_single_detailed(
        self,
        features: dict,
        feature_names: list[str],
        X_seq: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Predict P(UP) with full per-model breakdown and feature contributions.

        Returns:
            dict with keys:
              - prob: calibrated ensemble P(UP)
              - prob_lgbm: raw LightGBM P(UP) or None
              - prob_xgb: raw XGBoost P(UP) or None
              - prob_lstm: raw LSTM P(UP) or None
              - top_features_pro: list of (name, importance) supporting the direction
              - top_features_contra: list of (name, importance) against the direction
        """
        X = np.array([[features.get(f, 0.0) for f in feature_names]])

        # Per-model raw probabilities
        prob_lgbm = float(self.lgbm.predict_proba(X)[0]) if self.lgbm.is_fitted else None
        prob_xgb = float(self.xgb.predict_proba(X)[0]) if self.xgb.is_fitted else None
        prob_lstm = float(self.lstm.predict_proba(X_seq)[0]) if (self.lstm.is_fitted and X_seq is not None) else None

        # Ensemble probability
        ensemble_prob = self.predict_proba(X, X_seq)
        prob = float(ensemble_prob[0])

        # Feature contributions from tree models
        top_pro, top_contra = self._get_feature_contributions(
            features, feature_names, direction_up=(prob >= 0.5)
        )

        return {
            'prob': prob,
            'prob_lgbm': prob_lgbm,
            'prob_xgb': prob_xgb,
            'prob_lstm': prob_lstm,
            'top_features_pro': top_pro,
            'top_features_contra': top_contra,
        }

    def _get_feature_contributions(
        self,
        features: dict,
        feature_names: list[str],
        direction_up: bool,
        top_n: int = 10,
    ) -> tuple[list, list]:
        """
        Get top features supporting (pro) and opposing (contra) the predicted direction.

        Uses LightGBM feature_importances_ (gain-based) as primary source,
        falls back to XGBoost if LightGBM is unavailable.

        A feature is "pro" if:
          - importance is high AND
          - its value is above median (for UP direction) or below median (for DOWN)

        For simplicity, we use the feature value's z-score relative to zero:
          - Positive feature value + UP prediction = PRO
          - Negative feature value + UP prediction = CONTRA

        Returns:
            (top_pro, top_contra) — each a list of {name, importance, value}
        """
        importance_dict = {}

        if self.lgbm.is_fitted:
            importance_dict = self.lgbm.feature_importance()
        elif self.xgb.is_fitted:
            importance_dict = self.xgb.feature_importance()
        else:
            return [], []

        # Build scored list
        scored = []
        for name in feature_names:
            imp = importance_dict.get(name, 0.0)
            val = features.get(name, 0.0)
            scored.append({'name': name, 'importance': round(imp, 4), 'value': round(val, 6)})

        # Sort by importance descending
        scored.sort(key=lambda x: x['importance'], reverse=True)

        # Take top features by importance and classify as pro/contra
        top_features = scored[:top_n * 2]  # Take enough to fill both lists
        pro = []
        contra = []

        for feat in top_features:
            val = feat['value']
            # Heuristic: features whose value aligns with the predicted direction are "pro"
            # For return-based features, positive value + UP = pro
            # For mean-reverting features (like RSI), this is approximate but useful
            if direction_up:
                is_pro = val >= 0
            else:
                is_pro = val <= 0

            entry = {'name': feat['name'], 'importance': feat['importance'], 'value': feat['value']}
            if is_pro and len(pro) < top_n:
                pro.append(entry)
            elif not is_pro and len(contra) < top_n:
                contra.append(entry)

        return pro, contra

    # ────────────────────────────────────────────────────────────────────
    # Training (delegates to sub-models)
    # ────────────────────────────────────────────────────────────────────

    def fit_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        feature_names: list[str],
        seq_length: int = 30,
    ) -> dict:
        """
        Train all sub-models and calibrate.

        Data Leakage Prevention — Temporal ordering:
          1. X_train/y_train: oldest data (training)
          2. X_val/y_val:   next chunk (early stopping / validation)
          3. X_cal/y_cal:   next chunk (calibration — NEVER seen during training)

        Args:
            X_train, y_train: Training data (2D flat).
            X_val, y_val: Validation data (2D flat).
            X_cal, y_cal: Calibration data (2D flat, strictly after val).
            feature_names: List of feature names.
            seq_length: LSTM sequence length.

        Returns:
            dict with training metadata from all sub-models.
        """
        t0 = time.perf_counter()
        results = {}

        # 1. Train LightGBM
        logger.info("ensemble_training_lgbm")
        results['lgbm'] = self.lgbm.fit(
            X_train, y_train, X_val, y_val, feature_names
        )

        # 2. Train XGBoost
        logger.info("ensemble_training_xgb")
        results['xgb'] = self.xgb.fit(
            X_train, y_train, X_val, y_val, feature_names
        )

        # 3. Train LSTM (build sequences)
        logger.info("ensemble_training_lstm")
        X_train_seq, y_train_seq = LSTMModel.build_sequences(X_train, y_train, seq_length)
        X_val_seq, y_val_seq = LSTMModel.build_sequences(X_val, y_val, seq_length)
        if len(X_train_seq) > 0:
            results['lstm'] = self.lstm.fit(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq
            )

        # 4. Calibrate on held-out calibration set
        logger.info("ensemble_calibrating")
        raw_cal_probs = self.predict_proba(X_cal, calibrate=False)
        self.calibrator.fit(raw_cal_probs, y_cal)
        results['calibration'] = self.calibrator.stats()

        elapsed = time.perf_counter() - t0
        self._metadata = {
            'total_train_time_sec': round(elapsed, 2),
            'weights': self.weights,
            'sub_models': results,
        }

        logger.info("ensemble_trained", elapsed_sec=round(elapsed, 2))
        return self._metadata

    # ────────────────────────────────────────────────────────────────────
    # Serialisation
    # ────────────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save all sub-models, calibrator, and metadata."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self.lgbm.save(str(path / 'lgbm'))
        self.xgb.save(str(path / 'xgb'))
        self.lstm.save(str(path / 'lstm'))
        self.calibrator.save(str(path / 'calibrator'))

        with open(path / 'ensemble_metadata.json', 'w') as f:
            json.dump({
                'weights': self.weights,
                'metadata': self._metadata,
            }, f, indent=2)

        logger.info("ensemble_saved", path=str(path))

    @classmethod
    def load(cls, directory: str) -> 'EnsembleModel':
        """Load all sub-models and calibrator. Target: <30 seconds."""
        t0 = time.perf_counter()
        path = Path(directory)

        lgbm = LGBMModel.load(str(path / 'lgbm'))
        xgb_model = XGBModel.load(str(path / 'xgb'))

        meta_path = path / 'ensemble_metadata.json'
        weights = WEIGHTS.copy()
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            weights = meta.get('weights', weights)

        lstm = LSTMModel.load(str(path / 'lstm'))
        calibrator = PlattCalibrator.load(str(path / 'calibrator'))

        elapsed = time.perf_counter() - t0
        logger.info("ensemble_loaded", path=str(path), elapsed_sec=round(elapsed, 2))

        instance = cls(lgbm=lgbm, xgb_model=xgb_model, lstm=lstm,
                       calibrator=calibrator, weights=weights)
        return instance

    @property
    def is_fitted(self) -> bool:
        return self.lgbm.is_fitted or self.xgb.is_fitted

    def stats(self) -> dict:
        return {
            'weights': self.weights,
            'lgbm_fitted': self.lgbm.is_fitted,
            'xgb_fitted': self.xgb.is_fitted,
            'lstm_fitted': self.lstm.is_fitted,
            'calibrator_fitted': self.calibrator.is_fitted,
        }
