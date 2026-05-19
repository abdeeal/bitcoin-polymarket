"""
Model Calibration — Platt Scaling for probability calibration.

Calibration Strategy (PRD 6.6):
  - Method: Platt scaling (logistic sigmoid fit on raw model output)
  - Target: Expected Calibration Error (ECE) < 0.03
  - Fitted on HELD-OUT calibration set only (never training/val data)

Data Leakage Prevention:
  - Calibration set is strictly chronologically AFTER validation set
  - Calibration parameters (A, B) are learned only from calibration data
  - During walk-forward, calibration is re-fitted at each fold boundary

Why Platt Scaling:
  - Tree-based models (LightGBM, XGBoost) produce probabilities that are
    often not well-calibrated (systematic over/under-confidence)
  - Platt scaling learns P(Y=1|s) = 1/(1+exp(As+B)) where s is the raw score
  - Only 2 parameters → low variance, works well with small calibration sets
"""

import json
from typing import Optional
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Calibration bins for ECE computation
N_BINS = 10


class PlattCalibrator:
    """
    Platt scaling calibrator — maps raw model outputs to calibrated probabilities.

    Learns: P(Y=1|s) = 1 / (1 + exp(A*s + B))
    where s is the raw model score and A, B are learned parameters.
    """

    def __init__(self):
        self._A: float = -1.0  # slope (initialised to identity mapping)
        self._B: float = 0.0   # intercept
        self._is_fitted: bool = False
        self._ece: float = 1.0
        self._n_samples: int = 0

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit Platt scaling parameters on held-out calibration data.

        Uses Newton's method to find optimal A, B that minimise
        negative log-likelihood of the sigmoid model.

        Data Leakage: raw_probs MUST come from a calibration set
        that was NOT used during model training or validation.

        Args:
            raw_probs: 1D array of raw model probabilities.
            y_true:    1D array of true binary labels.
        """
        n = len(raw_probs)
        if n < 10:
            logger.warning("calibrator_insufficient_data", n=n)
            self._is_fitted = True
            return

        self._n_samples = n

        # Platt scaling target transformation
        prior1 = np.sum(y_true == 1)
        prior0 = n - prior1
        # Smoothed targets
        hi_target = (prior1 + 1) / (prior1 + 2)
        lo_target = 1 / (prior0 + 2)
        targets = np.where(y_true == 1, hi_target, lo_target)

        # Newton's method for A, B
        A = 0.0
        B = np.log((prior0 + 1) / (prior1 + 1))

        for iteration in range(100):
            # Compute sigmoid
            s = A * raw_probs + B
            p = 1.0 / (1.0 + np.exp(-s))
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # Gradient
            d1 = targets - p
            d2 = p * (1 - p)

            # Hessian components
            g_A = np.sum(d1 * raw_probs)
            g_B = np.sum(d1)
            h_AA = -np.sum(d2 * raw_probs * raw_probs)
            h_BB = -np.sum(d2)
            h_AB = -np.sum(d2 * raw_probs)

            det = h_AA * h_BB - h_AB * h_AB
            if abs(det) < 1e-10:
                break

            # Newton step
            dA = -(h_BB * g_A - h_AB * g_B) / det
            dB = -(h_AA * g_B - h_AB * g_A) / det

            A += dA
            B += dB

            if abs(dA) < 1e-8 and abs(dB) < 1e-8:
                break

        self._A = A
        self._B = B
        self._is_fitted = True

        # Compute calibrated probs and ECE
        cal_probs = self.transform(raw_probs)
        self._ece = self._compute_ece(cal_probs, y_true)

        logger.info(
            "calibrator_fitted",
            A=round(A, 4),
            B=round(B, 4),
            ECE=round(self._ece, 4),
            n_samples=n,
        )

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities."""
        s = self._A * raw_probs + self._B
        return 1.0 / (1.0 + np.exp(-s))

    @staticmethod
    def _compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = N_BINS) -> float:
        """
        Compute Expected Calibration Error.

        ECE = sum_{b=1}^{B} (|S_b|/n) × |acc(S_b) - conf(S_b)|

        where S_b is the set of samples in bin b.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(probs)

        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            n_bin = np.sum(mask)
            if n_bin == 0:
                continue
            acc = np.mean(y_true[mask])
            conf = np.mean(probs[mask])
            ece += (n_bin / n) * abs(acc - conf)

        return float(ece)

    # ────────────────────────────────────────────────────────────────────
    # Serialisation
    # ────────────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'calibrator.json', 'w') as f:
            json.dump({
                'A': self._A, 'B': self._B,
                'is_fitted': self._is_fitted,
                'ece': self._ece,
                'n_samples': self._n_samples,
            }, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> 'PlattCalibrator':
        path = Path(directory)
        instance = cls()
        cal_path = path / 'calibrator.json'
        if cal_path.exists():
            with open(cal_path) as f:
                data = json.load(f)
            instance._A = data['A']
            instance._B = data['B']
            instance._is_fitted = data['is_fitted']
            instance._ece = data.get('ece', 1.0)
            instance._n_samples = data.get('n_samples', 0)
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def stats(self) -> dict:
        return {
            'is_fitted': self._is_fitted,
            'A': round(self._A, 6),
            'B': round(self._B, 6),
            'ECE': round(self._ece, 4),
            'n_samples': self._n_samples,
        }
