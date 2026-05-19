"""
Walk-Forward Validation — Time-series cross-validation with data leakage prevention.

Protocol (PRD 6.4):
  Training window : Rolling 90-day window
  Validation       : Next 10 days (OOS, strict chronological)
  Calibration      : Next 5 days (held-out, after validation)
  Step             : Retrain every 10 days
  Minimum bars     : 2,592 (90 days × 288 bars/day at 5-min)

Data Leakage Prevention Design:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  DATA LEAKAGE PREVENTION — CRITICAL INVARIANTS                    │
  │                                                                   │
  │  1. TEMPORAL ORDERING: train_end < val_start < cal_start < test   │
  │     - No future data EVER leaks into training                     │
  │     - Validation is strictly after training                       │
  │     - Calibration is strictly after validation                    │
  │                                                                   │
  │  2. FEATURE COMPUTATION: features computed BEFORE label creation  │
  │     - Features at time T use only data from [T-window, T]         │
  │     - Label at T is close[T+5] > close[T] — future-looking BY    │
  │       DESIGN (it's the target, not a feature)                     │
  │     - Features NEVER include any data from [T+1, T+5]            │
  │                                                                   │
  │  3. NORMALISATION: scaler fit on training set ONLY                │
  │     - Mean/std from training set applied to val, cal, and test    │
  │     - No global normalisation across entire dataset               │
  │                                                                   │
  │  4. PURGING: gap between train and val to prevent autocorrelation │
  │     - 5-bar (25 min) purge gap eliminates label overlap           │
  │     - Without this, last training label overlaps with first val   │
  │                                                                   │
  │  5. EMBARGO: no samples from within the target horizon of the     │
  │     last training sample are included in validation               │
  │     - Target horizon = 5 bars (25 minutes)                        │
  └─────────────────────────────────────────────────────────────────────┘

Walk-Forward Timeline per Fold:
  |--- TRAIN (90 days) ---|--PURGE--|--- VAL (10 days) ---|--- CAL (5 days) ---|
  |                       | 5 bars  |                     |                     |
  | Features + Labels     | gap     | OOS evaluation      | Calibration only    |
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Walk-forward constants (in bars)
BARS_PER_DAY = 288          # 24h × 60min / 5min per bar (5-min candles)
TRAIN_DAYS = 90
VAL_DAYS = 10
CAL_DAYS = 5
STEP_DAYS = 10
PURGE_BARS = 5              # 25 minutes — eliminates label overlap
LABEL_HORIZON = 5           # target: close[T+5] vs close[T]


@dataclass
class WalkForwardFold:
    """One fold of walk-forward validation."""
    fold_id: int
    train_start: int            # index into full dataset
    train_end: int
    val_start: int
    val_end: int
    cal_start: int
    cal_end: int

    # Results (filled after training)
    train_samples: int = 0
    val_samples: int = 0
    cal_samples: int = 0
    val_accuracy: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    ece: float = 0.0
    train_time_sec: float = 0.0
    val_probs: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    def validate_no_leakage(self) -> None:
        """Assert temporal ordering invariants."""
        assert self.train_end < self.val_start, \
            f"LEAKAGE: train_end ({self.train_end}) >= val_start ({self.val_start})"
        assert self.val_end < self.cal_start, \
            f"LEAKAGE: val_end ({self.val_end}) >= cal_start ({self.cal_start})"
        assert self.val_start - self.train_end >= PURGE_BARS, \
            f"LEAKAGE: purge gap ({self.val_start - self.train_end}) < {PURGE_BARS}"


@dataclass
class WalkForwardResult:
    """Aggregated results across all folds."""
    folds: list[WalkForwardFold] = field(default_factory=list)
    total_time_sec: float = 0.0

    @property
    def mean_accuracy(self) -> float:
        accs = [f.val_accuracy for f in self.folds if f.val_accuracy > 0]
        return float(np.mean(accs)) if accs else 0.0

    @property
    def mean_logloss(self) -> float:
        losses = [f.val_logloss for f in self.folds if f.val_logloss > 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def mean_auc(self) -> float:
        aucs = [f.val_auc for f in self.folds if f.val_auc > 0]
        return float(np.mean(aucs)) if aucs else 0.0

    @property
    def mean_ece(self) -> float:
        eces = [f.ece for f in self.folds if f.ece > 0]
        return float(np.mean(eces)) if eces else 0.0

    def summary(self) -> dict:
        return {
            'n_folds': len(self.folds),
            'mean_accuracy': round(self.mean_accuracy, 4),
            'mean_logloss': round(self.mean_logloss, 4),
            'mean_auc': round(self.mean_auc, 4),
            'mean_ece': round(self.mean_ece, 4),
            'total_time_sec': round(self.total_time_sec, 1),
            'folds': [
                {
                    'fold': f.fold_id,
                    'train': f'{f.train_start}-{f.train_end}',
                    'val': f'{f.val_start}-{f.val_end}',
                    'cal': f'{f.cal_start}-{f.cal_end}',
                    'accuracy': round(f.val_accuracy, 4),
                    'log_loss': round(f.val_logloss, 4),
                    'auc': round(f.val_auc, 4),
                    'ece': round(f.ece, 4),
                }
                for f in self.folds
            ],
        }


class WalkForwardValidator:
    """
    Walk-forward cross-validation engine with built-in leakage prevention.

    Generates chronological train/val/cal splits from a dataset,
    trains the ensemble on each fold, and aggregates OOS metrics.
    """

    def __init__(
        self,
        train_days: int = TRAIN_DAYS,
        val_days: int = VAL_DAYS,
        cal_days: int = CAL_DAYS,
        step_days: int = STEP_DAYS,
        purge_bars: int = PURGE_BARS,
        bars_per_day: int = BARS_PER_DAY,
    ):
        self._train_bars = train_days * bars_per_day
        self._val_bars = val_days * bars_per_day
        self._cal_bars = cal_days * bars_per_day
        self._step_bars = step_days * bars_per_day
        self._purge_bars = purge_bars
        self._bars_per_day = bars_per_day

    def generate_folds(self, n_samples: int) -> list[WalkForwardFold]:
        """
        Generate walk-forward fold indices.

        Each fold maintains strict temporal ordering:
          train → purge_gap → validation → calibration

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of WalkForwardFold with index ranges.
        """
        folds = []
        fold_id = 0

        # Minimum dataset size
        min_required = self._train_bars + self._purge_bars + self._val_bars + self._cal_bars
        if n_samples < min_required:
            logger.warning(
                "walk_forward_insufficient_data",
                n_samples=n_samples,
                required=min_required,
            )
            return []

        train_start = 0
        while True:
            train_end = train_start + self._train_bars - 1
            val_start = train_end + 1 + self._purge_bars  # purge gap
            val_end = val_start + self._val_bars - 1
            cal_start = val_end + 1
            cal_end = cal_start + self._cal_bars - 1

            if cal_end >= n_samples:
                break

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                cal_start=cal_start,
                cal_end=cal_end,
            )

            # Verify no data leakage
            fold.validate_no_leakage()
            folds.append(fold)

            # Step forward
            train_start += self._step_bars
            fold_id += 1

        logger.info(
            "walk_forward_folds_generated",
            n_folds=len(folds),
            n_samples=n_samples,
        )
        return folds

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ensemble,
        feature_names: list[str],
        seq_length: int = 30,
    ) -> WalkForwardResult:
        """
        Execute walk-forward validation across all folds.

        Args:
            X: 2D feature array (n_samples, n_features), chronological.
            y: 1D label array (n_samples,).
            ensemble: EnsembleModel instance.
            feature_names: Feature name list.
            seq_length: LSTM sequence length.

        Returns:
            WalkForwardResult with per-fold and aggregated metrics.
        """
        t0 = time.perf_counter()
        folds = self.generate_folds(len(X))

        if not folds:
            logger.error("walk_forward_no_folds")
            return WalkForwardResult()

        result = WalkForwardResult()

        for fold in folds:
            logger.info(
                "walk_forward_fold_start",
                fold=fold.fold_id,
                train=f"{fold.train_start}-{fold.train_end}",
                val=f"{fold.val_start}-{fold.val_end}",
            )

            fold_t0 = time.perf_counter()

            # Extract splits
            X_train = X[fold.train_start:fold.train_end + 1]
            y_train = y[fold.train_start:fold.train_end + 1]
            X_val = X[fold.val_start:fold.val_end + 1]
            y_val = y[fold.val_start:fold.val_end + 1]
            X_cal = X[fold.cal_start:fold.cal_end + 1]
            y_cal = y[fold.cal_start:fold.cal_end + 1]

            # Remove NaN labels (from shift(-5) at edges)
            train_mask = ~np.isnan(y_train)
            val_mask = ~np.isnan(y_val)
            cal_mask = ~np.isnan(y_cal)

            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_val[val_mask], y_val[val_mask]
            X_cal, y_cal = X_cal[cal_mask], y_cal[cal_mask]

            if len(y_train) < 100 or len(y_val) < 10:
                logger.warning("walk_forward_fold_skipped", fold=fold.fold_id)
                continue

            # Train ensemble
            ensemble.fit_all(
                X_train, y_train,
                X_val, y_val,
                X_cal, y_cal,
                feature_names,
                seq_length,
            )

            # Evaluate on validation set (with calibration)
            val_probs = ensemble.predict_proba(X_val)
            val_preds = (val_probs >= 0.5).astype(int)

            # Metrics
            fold.train_samples = len(y_train)
            fold.val_samples = len(y_val)
            fold.cal_samples = len(y_cal)
            fold.val_accuracy = float(np.mean(val_preds == y_val))
            fold.val_logloss = self._logloss(y_val, val_probs)
            fold.val_auc = self._auc(y_val, val_probs)
            fold.ece = self._ece(y_val, val_probs)
            fold.train_time_sec = time.perf_counter() - fold_t0
            fold.val_probs = val_probs
            fold.y_val = y_val

            logger.info(
                "walk_forward_fold_complete",
                fold=fold.fold_id,
                accuracy=round(fold.val_accuracy, 4),
                logloss=round(fold.val_logloss, 4),
                auc=round(fold.val_auc, 4),
                ece=round(fold.ece, 4),
            )

            result.folds.append(fold)

        result.total_time_sec = time.perf_counter() - t0

        logger.info(
            "walk_forward_complete",
            n_folds=len(result.folds),
            mean_accuracy=round(result.mean_accuracy, 4),
            mean_logloss=round(result.mean_logloss, 4),
            total_sec=round(result.total_time_sec, 1),
        )

        return result

    # ────────────────────── Metrics ──────────────────────────────────

    @staticmethod
    def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    @staticmethod
    def _auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simple AUC computation without sklearn."""
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        n_correct = 0
        total = len(pos) * len(neg)
        for p in pos:
            n_correct += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(n_correct / total)

    @staticmethod
    def _ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_pred)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_pred >= lo) & (y_pred < hi if i < n_bins - 1 else y_pred <= hi)
            if np.sum(mask) == 0:
                continue
            acc = np.mean(y_true[mask])
            conf = np.mean(y_pred[mask])
            ece += (np.sum(mask) / n) * abs(acc - conf)
        return float(ece)
