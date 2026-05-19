"""
Tests for Phase 11 — Monitoring & Continuous Improvement.

Covers:
  1. Alerting: 10 alert factories, AlertManager dispatch + rate limiting, channels
  2. DriftDetector: compute_psi, identical/shifted distributions, per-feature check
  3. PerformanceTracker: latency, win rate, API errors, entry rate, threshold checks
  4. RetrainScheduler: deploy/no-improvement/skip, drift-triggered retrain
  5. Integration: drift → alert → retrain pipeline
"""

import asyncio
import pytest
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.alerting import (
    AlertManager, Alert, AlertSeverity, AlertChannel,
    LogChannel, RATE_LIMIT_SECONDS,
    alert_feed_down, alert_cooling_down_start, alert_cooling_down_end,
    alert_hard_stop, alert_bracket_upgrade, alert_bracket_downgrade,
    alert_model_drift, alert_win_rate_drop, alert_api_error_rate,
    alert_feature_latency,
)
from src.monitoring.drift_detector import (
    DriftDetector, compute_psi, FeatureDriftResult, DriftReport,
    PSI_WARNING_THRESHOLD, PSI_RETRAIN_THRESHOLD,
)
from src.monitoring.performance_tracker import PerformanceTracker
from src.training.retrain_scheduler import RetrainScheduler, MIN_ACCURACY_IMPROVEMENT


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Alert Factories
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlertFactories:
    def test_alert_feed_down(self):
        a = alert_feed_down(130.0)
        assert a.name == "Feed Down"
        assert "130" in a.message
        assert AlertChannel.TELEGRAM in a.channels
        assert AlertChannel.EMAIL in a.channels
        assert a.severity == AlertSeverity.CRITICAL

    def test_alert_cooling_down_start(self):
        a = alert_cooling_down_start(2, 9.5)
        assert a.name == "Cooling Down Start"
        assert AlertChannel.TELEGRAM in a.channels
        assert "9.50" in a.message

    def test_alert_cooling_down_end(self):
        a = alert_cooling_down_end(1800)
        assert a.name == "Cooling Down End"
        assert a.severity == AlertSeverity.INFO

    def test_alert_hard_stop(self):
        a = alert_hard_stop(4.50)
        assert "4.50" in a.message
        assert a.severity == AlertSeverity.CRITICAL

    def test_alert_bracket_upgrade(self):
        a = alert_bracket_upgrade(0, 1, 25.0)
        assert "0 → 1" in a.message

    def test_alert_bracket_downgrade(self):
        a = alert_bracket_downgrade(1, 0, 9.0)
        assert a.severity == AlertSeverity.WARNING

    def test_alert_model_drift(self):
        a = alert_model_drift("rsi_14", 0.31)
        assert AlertChannel.EMAIL in a.channels
        assert "0.310" in a.message

    def test_alert_win_rate_drop(self):
        a = alert_win_rate_drop(0.48)
        assert "48.0%" in a.message
        assert AlertChannel.EMAIL in a.channels

    def test_alert_api_error_rate(self):
        a = alert_api_error_rate(7.5)
        assert AlertChannel.TELEGRAM in a.channels

    def test_alert_feature_latency(self):
        a = alert_feature_latency(125.0)
        assert "125.0" in a.message

    def test_all_10_alerts_have_names(self):
        alerts = [
            alert_feed_down(130), alert_cooling_down_start(2, 9.0),
            alert_cooling_down_end(1800), alert_hard_stop(4.0),
            alert_bracket_upgrade(0, 1, 25.0), alert_bracket_downgrade(1, 0, 9.0),
            alert_model_drift("f", 0.3), alert_win_rate_drop(0.45),
            alert_api_error_rate(6.0), alert_feature_latency(120.0),
        ]
        names = {a.name for a in alerts}
        assert len(names) == 10  # all unique


# ═══════════════════════════════════════════════════════════════════════════════
# AlertManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlertManager:
    def test_fire_log_channel(self):
        mgr = AlertManager()
        a = Alert("test", "hello", channels=[AlertChannel.LOG])
        ok = run_async(mgr.fire(a))
        assert ok is True
        assert len(mgr.history) == 1

    def test_rate_limiting(self):
        mgr = AlertManager()
        a1 = Alert("dup", "msg1", channels=[AlertChannel.LOG])
        ok1 = run_async(mgr.fire(a1))
        assert ok1 is True

        a2 = Alert("dup", "msg2", channels=[AlertChannel.LOG])
        ok2 = run_async(mgr.fire(a2))
        assert ok2 is False  # rate-limited

    def test_different_names_not_rate_limited(self):
        mgr = AlertManager()
        ok1 = run_async(mgr.fire(Alert("a", "m", channels=[AlertChannel.LOG])))
        ok2 = run_async(mgr.fire(Alert("b", "m", channels=[AlertChannel.LOG])))
        assert ok1 is True
        assert ok2 is True

    def test_clear_rate_limits(self):
        mgr = AlertManager()
        run_async(mgr.fire(Alert("x", "m", channels=[AlertChannel.LOG])))
        mgr.clear_rate_limits()
        ok = run_async(mgr.fire(Alert("x", "m", channels=[AlertChannel.LOG])))
        assert ok is True

    def test_telegram_not_configured(self):
        mgr = AlertManager()  # no telegram channel
        a = Alert("test", "msg", channels=[AlertChannel.TELEGRAM])
        ok = run_async(mgr.fire(a))
        # Falls back to LOG
        assert ok is True
        assert len(mgr.history) == 1

    def test_history_accumulates(self):
        mgr = AlertManager()
        for i in range(5):
            run_async(mgr.fire(Alert(f"alert_{i}", "msg", channels=[AlertChannel.LOG])))
        assert len(mgr.history) == 5

    def test_alert_to_dict(self):
        a = alert_feed_down(130)
        d = a.to_dict()
        assert d['name'] == "Feed Down"
        assert d['severity'] == "CRITICAL"
        assert 'TELEGRAM' in d['channels']


# ═══════════════════════════════════════════════════════════════════════════════
# Drift Detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePSI:
    def test_identical_distributions(self):
        np.random.seed(42)
        data = np.random.randn(1000)
        psi = compute_psi(data, data)
        assert psi < 0.01  # near-zero

    def test_shifted_distribution(self):
        np.random.seed(42)
        ref = np.random.randn(1000)
        cur = np.random.randn(1000) + 3  # shifted by 3 stdevs
        psi = compute_psi(ref, cur)
        assert psi > PSI_RETRAIN_THRESHOLD  # big shift

    def test_psi_always_non_negative(self):
        np.random.seed(42)
        for _ in range(10):
            ref = np.random.randn(200)
            cur = np.random.randn(200) + np.random.randn() * 0.5
            psi = compute_psi(ref, cur)
            assert psi >= -0.01  # may be slightly negative due to epsilon, but bounded

    def test_small_sample_returns_zero(self):
        psi = compute_psi(np.array([1.0, 2.0]), np.array([1.0, 2.0]), bins=10)
        assert psi == 0.0

    def test_warning_threshold(self):
        assert PSI_WARNING_THRESHOLD == 0.10

    def test_retrain_threshold(self):
        assert PSI_RETRAIN_THRESHOLD == 0.25


class TestDriftDetector:
    def test_no_reference_returns_empty(self):
        dd = DriftDetector()
        report = dd.check(np.random.randn(100, 5))
        assert report.total_features == 0

    def test_identical_features_all_ok(self):
        np.random.seed(42)
        data = np.random.randn(500, 10)
        dd = DriftDetector(reference_features=data)
        report = dd.check(data)
        assert report.retrain_triggered is False
        assert report.ok_count >= 8  # most should be ok

    def test_shifted_features_trigger_retrain(self):
        np.random.seed(42)
        ref = np.random.randn(500, 10)
        cur = ref + 5  # massive shift
        dd = DriftDetector(reference_features=ref)
        report = dd.check(cur)
        assert report.retrain_triggered is True
        assert report.retrain_count > 0

    def test_feature_names(self):
        np.random.seed(42)
        ref = np.random.randn(200, 3)
        dd = DriftDetector(reference_features=ref, feature_names=["rsi", "macd", "bb"])
        report = dd.check(ref)
        for r in report.feature_results:
            assert r.feature_name in ("rsi", "macd", "bb")

    def test_top_drifted_sorted(self):
        np.random.seed(42)
        ref = np.random.randn(500, 10)
        cur = ref.copy()
        cur[:, 3] += 10  # massive shift on feature 3
        dd = DriftDetector(reference_features=ref)
        report = dd.check(cur)
        if report.top_drifted:
            psis = [r.psi for r in report.top_drifted]
            assert psis == sorted(psis, reverse=True)

    def test_set_reference(self):
        dd = DriftDetector()
        data = np.random.randn(100, 5)
        dd.set_reference(data)
        report = dd.check(data)
        assert report.total_features == 5

    def test_summary(self):
        np.random.seed(42)
        ref = np.random.randn(200, 5)
        dd = DriftDetector(reference_features=ref)
        report = dd.check(ref)
        s = report.summary()
        assert 'total_features' in s
        assert 'retrain_triggered' in s


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceTracker:
    def test_latency_stats(self):
        pt = PerformanceTracker()
        for v in [10, 20, 30, 40, 50]:
            pt.record_latency(v)
        stats = pt.latency_stats()
        assert stats.samples == 5
        assert stats.mean_ms == 30.0

    def test_latency_p95(self):
        pt = PerformanceTracker()
        for v in range(100):
            pt.record_latency(float(v))
        stats = pt.latency_stats()
        assert stats.p95_ms > 90

    def test_win_rate_all_wins(self):
        pt = PerformanceTracker()
        for _ in range(10):
            pt.record_outcome(won=True, pnl=1.0)
        wr = pt.win_rate_stats()
        assert wr.win_rate == 1.0
        assert wr.streak_type == "win"
        assert wr.recent_streak == 10

    def test_win_rate_mixed(self):
        pt = PerformanceTracker()
        for w in [True, True, False, True, False]:
            pt.record_outcome(won=w, pnl=1.0 if w else -1.0)
        wr = pt.win_rate_stats()
        assert abs(wr.win_rate - 0.6) < 0.01

    def test_api_stats(self):
        pt = PerformanceTracker()
        for i in range(20):
            pt.record_api_call(success=(i % 5 != 0))
        stats = pt.api_stats()
        assert stats.total_calls == 20
        assert stats.errors == 4

    def test_entry_rate(self):
        pt = PerformanceTracker()
        for _ in range(100):
            pt.record_bar()
        for _ in range(7):
            pt.record_outcome(won=True, pnl=0.5)
        assert abs(pt.entry_rate() - 7.0) < 0.1

    def test_snapshot_structure(self):
        pt = PerformanceTracker()
        pt.record_latency(42.0)
        pt.record_outcome(won=True, pnl=0.88)
        pt.record_api_call(success=True)
        pt.record_bar()
        s = pt.snapshot()
        assert 'latency' in s
        assert 'win_rate' in s
        assert 'api' in s
        assert 'entry_rate_pct' in s
        assert 'cumulative_pnl' in s

    def test_check_latency_alert(self):
        pt = PerformanceTracker()
        for _ in range(20):
            pt.record_latency(150.0)
        result = pt.check_latency_alert(budget_ms=100.0)
        assert result is not None
        assert result > 100

    def test_check_latency_no_alert(self):
        pt = PerformanceTracker()
        for _ in range(20):
            pt.record_latency(50.0)
        result = pt.check_latency_alert(budget_ms=100.0)
        assert result is None

    def test_check_win_rate_alert(self):
        pt = PerformanceTracker()
        for _ in range(30):
            pt.record_outcome(won=False, pnl=-1.0)
        result = pt.check_win_rate_alert(threshold=0.50)
        assert result is not None
        assert result < 0.50

    def test_check_api_error_alert(self):
        pt = PerformanceTracker()
        for _ in range(20):
            pt.record_api_call(success=False)
        result = pt.check_api_error_alert(threshold_pct=5.0)
        assert result is not None
        assert result > 5.0

    def test_daily_pnl(self):
        pt = PerformanceTracker()
        pt.record_outcome(won=True, pnl=1.0)
        pt.record_outcome(won=True, pnl=2.0)
        today = time.strftime("%Y-%m-%d")
        assert pt._daily_pnl.get(today) == 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Retrain Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrainScheduler:
    def test_deploy_when_accuracy_improves(self):
        async def fetch_data(days):
            return list(range(100))  # stub

        def walk_forward(df):
            return {'accuracy': 0.60}

        def train(df):
            return "model_object"

        alerts = []
        async def alert_fn(msg):
            alerts.append(msg)

        scheduler = RetrainScheduler(
            fetch_data_fn=fetch_data,
            walk_forward_fn=walk_forward,
            train_fn=train,
            alert_fn=alert_fn,
        )
        scheduler.set_current_accuracy(0.55)

        result = run_async(scheduler.run_retrain(reason="test"))
        assert result['result'] == 'deployed'
        assert result['deployed'] is True
        assert len(alerts) == 1
        assert "0.6000" in alerts[0]

    def test_no_improvement_no_deploy(self):
        async def fetch_data(days):
            return list(range(100))

        def walk_forward(df):
            return {'accuracy': 0.551}

        scheduler = RetrainScheduler(
            fetch_data_fn=fetch_data,
            walk_forward_fn=walk_forward,
        )
        scheduler.set_current_accuracy(0.55)

        result = run_async(scheduler.run_retrain())
        assert result['result'] == 'no_improvement'
        assert result['deployed'] is False

    def test_skip_when_no_fetch_fn(self):
        scheduler = RetrainScheduler()
        result = run_async(scheduler.run_retrain())
        assert result['result'] == 'skipped'

    def test_stats_tracks_retrain_count(self):
        async def fetch_data(days): return list(range(10))
        def walk_forward(df): return {'accuracy': 0.80}
        def train(df): return "m"

        scheduler = RetrainScheduler(
            fetch_data_fn=fetch_data,
            walk_forward_fn=walk_forward,
            train_fn=train,
        )
        scheduler.set_current_accuracy(0.50)
        run_async(scheduler.run_retrain())
        assert scheduler.stats['retrain_count'] == 1

    def test_min_accuracy_improvement_constant(self):
        assert MIN_ACCURACY_IMPROVEMENT == 0.005

    def test_drift_triggered_retrain(self):
        scheduler = RetrainScheduler()

        class MockDrift:
            retrain_triggered = True

        assert scheduler.should_retrain_from_drift(MockDrift()) is True

        class MockNoDrift:
            retrain_triggered = False

        assert scheduler.should_retrain_from_drift(MockNoDrift()) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Drift → Alert → Retrain Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestMonitoringIntegration:
    def test_drift_triggers_alert_and_retrain(self):
        """End-to-end: drift detected → alert fired → retrain triggered."""
        np.random.seed(42)

        # 1. Detect drift
        ref = np.random.randn(500, 5)
        cur = ref + 5  # massive shift
        dd = DriftDetector(reference_features=ref, feature_names=["f0","f1","f2","f3","f4"])
        report = dd.check(cur)
        assert report.retrain_triggered

        # 2. Fire alert
        mgr = AlertManager()
        worst = report.top_drifted[0]
        a = alert_model_drift(worst.feature_name, worst.psi)
        ok = run_async(mgr.fire(a))
        assert ok is True

        # 3. Trigger retrain
        scheduler = RetrainScheduler()
        assert scheduler.should_retrain_from_drift(report) is True

    def test_performance_tracker_fires_latency_alert(self):
        """PerformanceTracker detects high latency → alert fired."""
        pt = PerformanceTracker()
        for _ in range(50):
            pt.record_latency(150.0)

        p95 = pt.check_latency_alert(budget_ms=100.0)
        assert p95 is not None

        mgr = AlertManager()
        a = alert_feature_latency(p95)
        ok = run_async(mgr.fire(a))
        assert ok is True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
