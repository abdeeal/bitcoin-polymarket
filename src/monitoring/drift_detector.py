"""
Drift Detector — Feature distribution shift monitoring via PSI.

PRD 14.3:
  - PSI (Population Stability Index) per feature
  - > 0.10 → warning
  - > 0.25 → retrain trigger
  - Computed between reference (training) and current (live) distributions

PSI formula:
  PSI = Σ (P_current - P_reference) * ln(P_current / P_reference)

Usage:
    detector = DriftDetector(reference_features=X_train)
    report = detector.check(current_features=X_live)
    if report['retrain_triggered']:
        scheduler.trigger_retrain()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

PSI_WARNING_THRESHOLD = 0.10
PSI_RETRAIN_THRESHOLD = 0.25
DEFAULT_BINS = 10


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = DEFAULT_BINS) -> float:
    """
    Compute Population Stability Index between two distributions.

    PRD 14.3 exact formula:
      ref = histogram(reference, bins) / len(reference) + ε
      cur = histogram(current, bins) / len(current) + ε
      PSI = Σ (cur - ref) * ln(cur / ref)

    Args:
        reference: Reference distribution (training data).
        current:   Current distribution (live data).
        bins:      Number of histogram bins.

    Returns:
        PSI value. 0 = identical. >0.25 = significant drift.
    """
    if len(reference) < bins or len(current) < bins:
        return 0.0

    # Use reference breaks for both histograms (ensures alignment)
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    # Normalize to proportions, add epsilon for numerical stability
    eps = 1e-8
    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""
    feature_name: str
    psi: float
    status: str  # "ok" | "warning" | "retrain"

    @property
    def is_warning(self) -> bool:
        return self.psi > PSI_WARNING_THRESHOLD

    @property
    def is_retrain(self) -> bool:
        return self.psi > PSI_RETRAIN_THRESHOLD

    def to_dict(self) -> dict:
        return {
            'feature_name': self.feature_name,
            'psi': round(self.psi, 4),
            'status': self.status,
        }


@dataclass
class DriftReport:
    """Aggregate drift report across all features."""
    total_features: int = 0
    ok_count: int = 0
    warning_count: int = 0
    retrain_count: int = 0
    retrain_triggered: bool = False
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    top_drifted: list[FeatureDriftResult] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            'total_features': self.total_features,
            'ok': self.ok_count,
            'warnings': self.warning_count,
            'retrain_triggers': self.retrain_count,
            'retrain_triggered': self.retrain_triggered,
            'top_drifted': [f.to_dict() for f in self.top_drifted[:5]],
        }


class DriftDetector:
    """
    Monitor feature drift between training (reference) and live (current) data.

    Stores the training distribution and checks live data against it.
    Tracks per-feature PSI with warning/retrain thresholds.

    Args:
        reference_features: Training feature matrix, shape (n_train, n_features).
        feature_names:      Optional list of feature names.
        bins:               Histogram bins for PSI computation.
    """

    def __init__(
        self,
        reference_features: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        bins: int = DEFAULT_BINS,
    ):
        self._reference = reference_features
        self._feature_names = feature_names
        self._bins = bins

    def set_reference(self, reference_features: np.ndarray, feature_names: Optional[list[str]] = None) -> None:
        """Set/update the reference (training) distribution."""
        self._reference = reference_features
        if feature_names:
            self._feature_names = feature_names

    def check(self, current_features: np.ndarray) -> DriftReport:
        """
        Check current features against reference for drift.

        Args:
            current_features: Live feature matrix, shape (n_live, n_features).

        Returns:
            DriftReport with per-feature PSI and retrain decision.
        """
        report = DriftReport()

        if self._reference is None or current_features is None:
            logger.warning("drift_check_skipped", reason="no reference or current data")
            return report

        n_features = min(self._reference.shape[1], current_features.shape[1])
        report.total_features = n_features

        for j in range(n_features):
            ref_col = self._reference[:, j]
            cur_col = current_features[:, j]

            # Skip constant features
            if ref_col.std() < 1e-10 and cur_col.std() < 1e-10:
                report.ok_count += 1
                continue

            psi = compute_psi(ref_col, cur_col, bins=self._bins)
            name = self._feature_names[j] if self._feature_names and j < len(self._feature_names) else f"feature_{j}"

            if psi > PSI_RETRAIN_THRESHOLD:
                status = "retrain"
                report.retrain_count += 1
            elif psi > PSI_WARNING_THRESHOLD:
                status = "warning"
                report.warning_count += 1
            else:
                status = "ok"
                report.ok_count += 1

            result = FeatureDriftResult(feature_name=name, psi=psi, status=status)
            report.feature_results.append(result)

        # Top drifted features (sorted by PSI descending)
        report.top_drifted = sorted(report.feature_results, key=lambda r: r.psi, reverse=True)[:10]
        report.retrain_triggered = report.retrain_count > 0

        if report.retrain_triggered:
            logger.warning(
                "drift_retrain_triggered",
                retrain_features=report.retrain_count,
                worst_psi=report.top_drifted[0].psi if report.top_drifted else 0,
            )
        else:
            logger.info("drift_check_passed", warnings=report.warning_count)

        return report
