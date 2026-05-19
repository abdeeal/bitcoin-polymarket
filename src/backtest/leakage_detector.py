"""
Leakage Detector — Automated data leakage prevention checks.

PRD 11.3 Anti-Leakage Checklist:
  1. Labels use shift(-5) — future data never visible to features
  2. Normalization uses only data up to t
  3. Walk-forward splits strictly chronological
  4. Cooling down skips execution, not data collection

This module provides both static checks (code inspection) and
runtime checks (data validation) to detect leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LeakageCheck:
    """Result of a single leakage check."""
    name: str
    passed: bool = False
    details: str = ''
    severity: str = 'critical'  # critical | warning

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'passed': self.passed,
            'details': self.details,
            'severity': self.severity,
        }


@dataclass
class LeakageReport:
    """Aggregate leakage detection results."""
    all_passed: bool = False
    checks: list[LeakageCheck] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            'all_passed': self.all_passed,
            'total_checks': len(self.checks),
            'passed': sum(1 for c in self.checks if c.passed),
            'failed': [c.to_dict() for c in self.checks if not c.passed],
        }


class LeakageDetector:
    """
    Detects data leakage in feature matrices and train/val splits.

    Static checks validate code patterns. Runtime checks validate
    actual data arrays for future-data contamination.

    Usage:
        detector = LeakageDetector()
        report = detector.run_all(
            feature_matrix=X,
            labels=y,
            timestamps=ts,
            train_indices=train_idx,
            val_indices=val_idx,
        )
        if not report.all_passed:
            raise ValueError("Data leakage detected!")
    """

    def run_all(
        self,
        feature_matrix: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
    ) -> LeakageReport:
        """Run all leakage checks and return aggregate report."""
        report = LeakageReport()

        report.checks.append(
            self.check_label_shift(labels, feature_matrix)
        )
        report.checks.append(
            self.check_chronological_split(timestamps, train_indices, val_indices)
        )
        report.checks.append(
            self.check_no_future_features(feature_matrix, labels, timestamps)
        )
        report.checks.append(
            self.check_normalization_window(feature_matrix, timestamps, train_indices)
        )

        report.all_passed = all(c.passed for c in report.checks)

        if report.all_passed:
            logger.info("leakage_detection_passed", checks=len(report.checks))
        else:
            failed = [c.name for c in report.checks if not c.passed]
            logger.error("leakage_detected", failed=failed)

        return report

    def check_label_shift(
        self,
        labels: Optional[np.ndarray],
        features: Optional[np.ndarray],
    ) -> LeakageCheck:
        """
        Check 1: Labels should NOT be correlated with features at time t.

        If labels are derived from future prices (shift(-5)), they should
        have near-zero correlation with features at time t. High correlation
        (> 0.5) suggests label leakage into features.
        """
        check = LeakageCheck('label_shift', severity='critical')

        if labels is None or features is None:
            check.passed = True
            check.details = 'No data provided — skipped'
            return check

        # Check correlation of each feature with labels
        max_corr = 0.0
        max_feat_idx = 0
        n_features = features.shape[1] if features.ndim > 1 else 1

        for j in range(min(n_features, 50)):  # sample first 50
            col = features[:, j] if features.ndim > 1 else features
            valid = ~(np.isnan(col) | np.isnan(labels.astype(float)))
            if valid.sum() < 10:
                continue
            corr = abs(np.corrcoef(col[valid], labels[valid].astype(float))[0, 1])
            if corr > max_corr:
                max_corr = corr
                max_feat_idx = j

        if max_corr < 0.5:
            check.passed = True
            check.details = f'Max |corr| = {max_corr:.3f} (feature {max_feat_idx}) — no leakage'
        else:
            check.details = (
                f'Max |corr| = {max_corr:.3f} (feature {max_feat_idx}) — '
                f'possible label leakage (threshold: 0.5)'
            )

        return check

    def check_chronological_split(
        self,
        timestamps: Optional[np.ndarray],
        train_indices: Optional[np.ndarray],
        val_indices: Optional[np.ndarray],
    ) -> LeakageCheck:
        """
        Check 2: Validation set timestamps must be strictly after training set.

        Any overlap means future data leaks into training.
        """
        check = LeakageCheck('chronological_split', severity='critical')

        if timestamps is None or train_indices is None or val_indices is None:
            check.passed = True
            check.details = 'No split data provided — skipped'
            return check

        train_max_ts = timestamps[train_indices].max()
        val_min_ts = timestamps[val_indices].min()

        if val_min_ts > train_max_ts:
            check.passed = True
            check.details = (
                f'Val starts at {val_min_ts:.0f} > train ends at {train_max_ts:.0f} — '
                f'strictly chronological'
            )
        else:
            check.details = (
                f'OVERLAP: val starts at {val_min_ts:.0f} but train ends at {train_max_ts:.0f}'
            )

        return check

    def check_no_future_features(
        self,
        features: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        timestamps: Optional[np.ndarray],
    ) -> LeakageCheck:
        """
        Check 3: Features at time t should not contain info from t+1, t+2, etc.

        Heuristic: autocorrelation of feature-label product should decay.
        If it doesn't, future data may be embedded in features.
        """
        check = LeakageCheck('no_future_features', severity='critical')

        if features is None or labels is None:
            check.passed = True
            check.details = 'No data provided — skipped'
            return check

        # Simple test: feature at t should not predict label at t-1
        # (reverse causality test)
        n = len(labels)
        if n < 100 or features.ndim < 2:
            check.passed = True
            check.details = 'Insufficient data for future-feature test'
            return check

        # Compute correlation of features[t] with labels[t-5]
        # If > 0.3, features may contain future info
        shifted_labels = np.roll(labels.astype(float), 5)  # shift forward
        shifted_labels[:5] = np.nan

        max_reverse_corr = 0.0
        for j in range(min(features.shape[1], 30)):
            col = features[:, j]
            valid = ~(np.isnan(col) | np.isnan(shifted_labels))
            if valid.sum() < 50:
                continue
            rc = abs(np.corrcoef(col[valid], shifted_labels[valid])[0, 1])
            max_reverse_corr = max(max_reverse_corr, rc)

        if max_reverse_corr < 0.3:
            check.passed = True
            check.details = f'Max reverse corr = {max_reverse_corr:.3f} — no future leakage'
        else:
            check.details = f'Max reverse corr = {max_reverse_corr:.3f} — possible future info in features'

        return check

    def check_normalization_window(
        self,
        features: Optional[np.ndarray],
        timestamps: Optional[np.ndarray],
        train_indices: Optional[np.ndarray],
    ) -> LeakageCheck:
        """
        Check 4: Feature statistics (mean/std) should differ between early and late data.

        If normalization uses the full dataset, early and late feature
        distributions will be artificially similar (leakage via look-ahead).
        """
        check = LeakageCheck('normalization_window', severity='warning')

        if features is None or features.ndim < 2:
            check.passed = True
            check.details = 'No feature data — skipped'
            return check

        n = features.shape[0]
        if n < 200:
            check.passed = True
            check.details = 'Too few rows for normalization check'
            return check

        # Compare first 10% vs last 10% feature statistics
        first_n = max(n // 10, 10)
        last_n = max(n // 10, 10)

        first_means = features[:first_n].mean(axis=0)
        last_means = features[-last_n:].mean(axis=0)

        # If means are suspiciously identical, normalization may be global
        diff = np.abs(first_means - last_means)
        pct_identical = np.mean(diff < 1e-6) * 100

        if pct_identical < 80:
            check.passed = True
            check.details = (
                f'{pct_identical:.0f}% features identical between first/last 10% — '
                f'normalization appears local'
            )
        else:
            check.details = (
                f'{pct_identical:.0f}% features identical — '
                f'normalization may use global (look-ahead) statistics'
            )

        return check
