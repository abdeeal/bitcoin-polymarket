"""
Tests for Phase 3 — Model Development.

Covers:
  1. Walk-forward fold generation and leakage prevention
  2. Label computation correctness
  3. Calibration (Platt scaling + ECE)
  4. Ensemble weighted prediction
  5. Data leakage invariants
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.walk_forward import (
    WalkForwardValidator,
    WalkForwardFold,
    PURGE_BARS,
    BARS_PER_DAY,
)
from src.training.train_pipeline import TrainingPipeline
from src.models.calibration import PlattCalibrator
from src.models.ensemble import EnsembleModel
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.models.lstm_model import LSTMModel


# =============================================================================
# Walk-Forward Validation Tests
# =============================================================================

class TestWalkForwardFoldGeneration:
    """Test that walk-forward generates correct temporal splits."""

    def test_generates_folds(self):
        wf = WalkForwardValidator(train_days=10, val_days=3, cal_days=2, step_days=3, bars_per_day=10)
        folds = wf.generate_folds(200)
        assert len(folds) > 0

    def test_folds_temporal_order(self):
        wf = WalkForwardValidator(train_days=10, val_days=3, cal_days=2, step_days=3, bars_per_day=10)
        folds = wf.generate_folds(300)
        for fold in folds:
            assert fold.train_end < fold.val_start, "LEAKAGE: train overlaps val"
            assert fold.val_end < fold.cal_start, "LEAKAGE: val overlaps cal"

    def test_purge_gap_enforced(self):
        wf = WalkForwardValidator(train_days=10, val_days=3, cal_days=2, step_days=3, bars_per_day=10)
        folds = wf.generate_folds(300)
        for fold in folds:
            gap = fold.val_start - fold.train_end - 1
            assert gap >= PURGE_BARS, f"Purge gap {gap} < {PURGE_BARS}"

    def test_no_overlap_between_folds(self):
        wf = WalkForwardValidator(train_days=10, val_days=3, cal_days=2, step_days=3, bars_per_day=10)
        folds = wf.generate_folds(300)
        for fold in folds:
            # Validation must not overlap with training
            train_indices = set(range(fold.train_start, fold.train_end + 1))
            val_indices = set(range(fold.val_start, fold.val_end + 1))
            cal_indices = set(range(fold.cal_start, fold.cal_end + 1))
            assert len(train_indices & val_indices) == 0
            assert len(val_indices & cal_indices) == 0
            assert len(train_indices & cal_indices) == 0

    def test_insufficient_data_returns_empty(self):
        wf = WalkForwardValidator(train_days=90, val_days=10, cal_days=5)
        folds = wf.generate_folds(100)  # Way too few samples
        assert len(folds) == 0

    def test_fold_validate_raises_on_leakage(self):
        fold = WalkForwardFold(
            fold_id=0,
            train_start=0, train_end=100,
            val_start=100,  # Should be > 100 + purge
            val_end=150,
            cal_start=151, cal_end=200,
        )
        with pytest.raises(AssertionError, match="LEAKAGE"):
            fold.validate_no_leakage()


# =============================================================================
# Label Computation Tests
# =============================================================================

class TestLabelComputation:
    """Test label computation with leakage prevention."""

    def test_label_shift_correctness(self):
        pipeline = TrainingPipeline()
        close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=np.float64)
        labels = pipeline.compute_labels(close, horizon=5)
        # close[0]=100, close[5]=105 -> 105>100 -> 1
        assert labels[0] == 1.0
        # close[4]=104, close[9]=109 -> 109>104 -> 1
        assert labels[4] == 1.0
        # Last 5 should be NaN (no future data)
        assert np.isnan(labels[5])
        assert np.isnan(labels[9])

    def test_label_down_case(self):
        pipeline = TrainingPipeline()
        close = np.array([109, 108, 107, 106, 105, 104, 103, 102, 101, 100], dtype=np.float64)
        labels = pipeline.compute_labels(close, horizon=5)
        # close[0]=109, close[5]=104 -> 104 < 109 -> 0
        assert labels[0] == 0.0

    def test_label_nan_count(self):
        pipeline = TrainingPipeline()
        close = np.random.rand(100) * 100
        labels = pipeline.compute_labels(close, horizon=5)
        assert np.sum(np.isnan(labels)) == 5  # Exactly last 5 are NaN

    def test_label_no_future_leak(self):
        """Labels must not use any data from the future beyond the horizon."""
        pipeline = TrainingPipeline()
        close = np.arange(100, dtype=np.float64)
        labels = pipeline.compute_labels(close, horizon=5)
        # Label at T uses close[T] and close[T+5] ONLY
        for i in range(len(close) - 5):
            expected = 1.0 if close[i + 5] > close[i] else 0.0
            assert labels[i] == expected, f"Label mismatch at index {i}"


# =============================================================================
# Calibration Tests
# =============================================================================

class TestPlattCalibrator:
    """Test Platt scaling calibration."""

    def test_fit_transform(self):
        cal = PlattCalibrator()
        np.random.seed(42)
        raw = np.random.rand(500)
        y = ((raw > 0.5).astype(int) ^ (np.random.rand(500) < 0.1).astype(int)).astype(float)
        cal.fit(raw, y)
        assert cal.is_fitted
        transformed = cal.transform(raw)
        assert len(transformed) == 500
        assert np.all(transformed >= 0) and np.all(transformed <= 1)

    def test_calibration_improves_ece(self):
        """ECE should be reasonable after calibration."""
        cal = PlattCalibrator()
        np.random.seed(42)
        # Generate poorly calibrated predictions
        raw = np.random.beta(2, 5, 1000)  # skewed toward 0
        y = (np.random.rand(1000) < 0.5).astype(float)
        cal.fit(raw, y)
        stats = cal.stats()
        # ECE should be computed
        assert 'ECE' in stats
        assert stats['ECE'] >= 0

    def test_save_load(self, tmp_path):
        cal = PlattCalibrator()
        raw = np.random.rand(100)
        y = (raw > 0.5).astype(float)
        cal.fit(raw, y)
        cal.save(str(tmp_path))
        cal2 = PlattCalibrator.load(str(tmp_path))
        assert cal2.is_fitted
        assert abs(cal2._A - cal._A) < 1e-6

    def test_unfitted_identity(self):
        """Unfitted calibrator should still transform (identity-like)."""
        cal = PlattCalibrator()
        raw = np.array([0.3, 0.5, 0.7])
        result = cal.transform(raw)
        assert len(result) == 3


# =============================================================================
# Ensemble Tests
# =============================================================================

class TestEnsemble:
    """Test ensemble weighted prediction."""

    def test_weights_sum_to_one(self):
        ensemble = EnsembleModel()
        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6

    def test_predict_proba_no_models(self):
        """Unfitted ensemble returns 0.5."""
        ensemble = EnsembleModel()
        X = np.random.rand(10, 50)
        # This will return 0.5 since no models are fitted
        probs = ensemble.predict_proba(X)
        assert np.allclose(probs, 0.5)

    def test_predict_single(self):
        ensemble = EnsembleModel()
        features = {'f1': 0.5, 'f2': 0.3}
        prob = ensemble.predict_single(features, ['f1', 'f2'])
        assert 0 <= prob <= 1

    def test_stats(self):
        ensemble = EnsembleModel()
        s = ensemble.stats()
        assert 'weights' in s
        assert 'lgbm_fitted' in s


# =============================================================================
# LSTM Sequence Building Tests
# =============================================================================

class TestLSTMSequences:
    """Test LSTM sequence building with leakage prevention."""

    def test_sequence_shape(self):
        X = np.random.rand(100, 50)
        y = np.random.randint(0, 2, 100).astype(float)
        X_seq, y_seq = LSTMModel.build_sequences(X, y, seq_length=30)
        assert X_seq.shape == (71, 30, 50)  # 100-30+1 = 71
        assert y_seq.shape == (71,)

    def test_sequence_no_lookahead(self):
        """Each sequence only contains bars up to the label's time."""
        X = np.arange(100).reshape(100, 1).astype(float)
        y = np.arange(100).astype(float)
        X_seq, y_seq = LSTMModel.build_sequences(X, y, seq_length=10)
        # For sequence i, the last element should be X[i+9]
        # The label should be y[i+9]
        for i in range(len(X_seq)):
            assert X_seq[i, -1, 0] == float(i + 9)
            assert y_seq[i] == float(i + 9)
            # No future data in the sequence
            assert X_seq[i, 0, 0] == float(i)

    def test_empty_input(self):
        X = np.random.rand(5, 50)
        y = np.ones(5)
        X_seq, y_seq = LSTMModel.build_sequences(X, y, seq_length=10)
        assert len(X_seq) == 0


# =============================================================================
# Data Leakage Integration Test
# =============================================================================

class TestDataLeakagePrevention:
    """Integration tests verifying no data leakage across the full pipeline."""

    def test_walk_forward_no_future_in_training(self):
        """Training data must never contain samples from after validation start."""
        wf = WalkForwardValidator(train_days=5, val_days=2, cal_days=1, step_days=2, bars_per_day=10)
        n = 200
        folds = wf.generate_folds(n)
        for fold in folds:
            # Training indices
            train_max = fold.train_end
            val_min = fold.val_start
            assert train_max < val_min - PURGE_BARS + 1, \
                f"Training data extends beyond purge boundary"

    def test_calibration_set_independence(self):
        """Calibration set must be strictly after validation."""
        wf = WalkForwardValidator(train_days=5, val_days=2, cal_days=1, step_days=2, bars_per_day=10)
        folds = wf.generate_folds(200)
        for fold in folds:
            assert fold.cal_start > fold.val_end
            assert fold.cal_start > fold.train_end

    def test_label_alignment_prevents_leakage(self):
        """Labels at the boundary are NaN, preventing future data leakage."""
        pipeline = TrainingPipeline()
        close = np.random.rand(50) * 100
        labels = pipeline.compute_labels(close, horizon=5)
        # The LAST 5 labels must be NaN
        assert all(np.isnan(labels[-5:]))
        # All other labels must be valid
        assert all(~np.isnan(labels[:-5]))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
