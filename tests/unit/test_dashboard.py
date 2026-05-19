"""
Tests for Phase 10 — Web Dashboard.

Covers:
  1. queries.py — compute_streaks() vectorized (PRD spec)
  2. queries.py — _empty_streaks() structure
  3. FastAPI REST endpoints — stub responses (no DB)
  4. FastAPI endpoint shapes — required keys per panel
  5. WSBroadcaster — stub payload structure
  6. WSBroadcaster — client tracking
"""

import asyncio
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi.testclient import TestClient

from src.dashboard.queries import compute_streaks, _empty_streaks
from src.dashboard.ws_broadcaster import WSBroadcaster


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Streak Computation Tests (PRD 13.3 Panel 6 spec)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeStreaks:
    def test_all_wins(self):
        outcomes = np.ones(10, dtype=bool)
        pnls = np.ones(10) * 0.88
        result = compute_streaks(outcomes, pnls)
        assert result['max_win_streak'] == 10
        assert result['max_lose_streak'] == 0
        assert result['current_win_streak'] == 10
        assert result['current_lose_streak'] == 0
        assert result['total_wins'] == 10
        assert result['total_losses'] == 0

    def test_all_losses(self):
        outcomes = np.zeros(5, dtype=bool)
        pnls = np.full(5, -1.0005)
        result = compute_streaks(outcomes, pnls)
        assert result['max_lose_streak'] == 5
        assert result['max_win_streak'] == 0
        assert result['current_lose_streak'] == 5
        assert result['current_win_streak'] == 0

    def test_alternating_wl(self):
        outcomes = np.array([True, False, True, False, True], dtype=bool)
        pnls = np.array([0.88, -1.0, 0.88, -1.0, 0.88])
        result = compute_streaks(outcomes, pnls)
        assert result['max_win_streak'] == 1
        assert result['max_lose_streak'] == 1

    def test_streak_pattern_ww_ll_www(self):
        outcomes = np.array([1,1,0,0,1,1,1], dtype=bool)
        pnls = np.array([0.88,0.88,-1.0,-1.0,0.88,0.88,0.88])
        result = compute_streaks(outcomes, pnls)
        assert result['max_win_streak'] == 3
        assert result['max_lose_streak'] == 2
        assert result['current_win_streak'] == 3
        assert result['current_lose_streak'] == 0

    def test_win_rate_calculation(self):
        outcomes = np.array([True, True, False, True, False], dtype=bool)
        pnls = np.array([0.88, 0.88, -1.0, 0.88, -1.0])
        result = compute_streaks(outcomes, pnls)
        assert abs(result['win_rate'] - 0.6) < 0.001

    def test_profit_factor(self):
        outcomes = np.array([True, True, False], dtype=bool)
        pnls = np.array([2.0, 2.0, -1.0])
        result = compute_streaks(outcomes, pnls)
        assert abs(result['profit_factor'] - 4.0) < 0.01

    def test_avg_win_avg_loss(self):
        outcomes = np.array([True, False, True, False], dtype=bool)
        pnls = np.array([1.0, -2.0, 2.0, -3.0])
        result = compute_streaks(outcomes, pnls)
        assert abs(result['avg_win_usd'] - 1.5) < 0.001
        assert abs(result['avg_loss_usd'] - (-2.5)) < 0.001

    def test_largest_win_loss(self):
        outcomes = np.array([True, True, False, False], dtype=bool)
        pnls = np.array([1.0, 5.0, -1.0, -3.0])
        result = compute_streaks(outcomes, pnls)
        assert result['largest_win_usd'] == pytest.approx(5.0)
        assert result['largest_loss_usd'] == pytest.approx(-3.0)

    def test_single_bet_win(self):
        outcomes = np.array([True], dtype=bool)
        pnls = np.array([0.88])
        result = compute_streaks(outcomes, pnls)
        assert result['total_entries'] == 1
        assert result['total_wins'] == 1
        assert result['total_losses'] == 0
        assert result['current_win_streak'] == 1

    def test_empty_streaks_keys(self):
        empty = _empty_streaks()
        required_keys = [
            'total_entries', 'total_wins', 'total_losses', 'win_rate',
            'max_win_streak', 'max_lose_streak', 'current_win_streak',
            'current_lose_streak', 'avg_win_usd', 'avg_loss_usd',
            'profit_factor', 'largest_win_usd', 'largest_loss_usd',
        ]
        for k in required_keys:
            assert k in empty, f"Missing key: {k}"

    def test_empty_streaks_all_zero(self):
        empty = _empty_streaks()
        assert all(v == 0 for v in empty.values())


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Endpoint Tests (stub/no-DB mode)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastAPIEndpoints:
    @pytest.fixture(autouse=True)
    def client(self):
        # Import app — pool will be None (no DB in test env)
        from src.dashboard.api import app
        self._client = TestClient(app, raise_server_exceptions=False)

    def test_health_endpoint(self):
        r = self._client.get('/health')
        assert r.status_code == 200
        data = r.json()
        assert 'status' in data
        assert data['status'] == 'ok'

    def test_summary_endpoint(self):
        r = self._client.get('/api/stats/summary')
        assert r.status_code == 200
        data = r.json()
        assert 'status' in data
        assert 'balance' in data
        assert 'pnl' in data

    def test_summary_balance_keys(self):
        r = self._client.get('/api/stats/summary')
        balance = r.json()['balance']
        required = ['starting_balance', 'current_balance', 'total_growth_usd', 'total_growth_pct']
        for k in required:
            assert k in balance, f"Missing: {k}"

    def test_summary_pnl_keys(self):
        r = self._client.get('/api/stats/summary')
        pnl = r.json()['pnl']
        required = ['total_pnl_gross', 'total_fees', 'total_pnl_net']
        for k in required:
            assert k in pnl, f"Missing: {k}"

    def test_period_endpoint(self):
        r = self._client.get('/api/stats/period')
        assert r.status_code == 200
        data = r.json()
        required = ['pnl_today_usd', 'pnl_week_usd', 'pnl_month_usd']
        for k in required:
            assert k in data

    def test_equity_curve_endpoint(self):
        r = self._client.get('/api/stats/equity-curve')
        assert r.status_code == 200
        data = r.json()
        assert 'equity_curve' in data
        assert 'cooling_down_periods' in data

    def test_equity_curve_days_param(self):
        r = self._client.get('/api/stats/equity-curve?days=7')
        assert r.status_code == 200

    def test_streaks_endpoint(self):
        r = self._client.get('/api/stats/streaks')
        assert r.status_code == 200
        data = r.json()
        assert 'total_entries' in data

    def test_by_side_endpoint(self):
        r = self._client.get('/api/stats/by-side')
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_by_hour_endpoint(self):
        r = self._client.get('/api/stats/by-hour')
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_by_day_endpoint(self):
        r = self._client.get('/api/stats/by-day')
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_calendar_endpoint(self):
        r = self._client.get('/api/stats/by-calendar?year=2026&month=5')
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_system_status_endpoint(self):
        r = self._client.get('/api/system/status')
        assert r.status_code == 200
        data = r.json()
        assert 'bot_state' in data  # stub returns bot_state, not uptime_formatted

    def test_calendar_invalid_month(self):
        r = self._client.get('/api/stats/by-calendar?year=2026&month=13')
        assert r.status_code == 422  # validation error

    def test_equity_invalid_days(self):
        r = self._client.get('/api/stats/equity-curve?days=0')
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# WSBroadcaster Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWSBroadcaster:
    def test_stub_payload_structure(self):
        broadcaster = WSBroadcaster(pool=None)
        payload = broadcaster._stub_payload()
        required = [
            'ts', 'bot_state', 'current_balance', 'consecutive_losses',
            'cooling_down_active', 'cooling_down_remaining_seconds',
            'last_bet', 'today_pnl', 'current_win_streak', 'current_lose_streak',
        ]
        for k in required:
            assert k in payload, f"Missing: {k}"

    def test_stub_payload_last_bet_keys(self):
        broadcaster = WSBroadcaster(pool=None)
        payload = broadcaster._stub_payload()
        lb = payload['last_bet']
        for k in ['timestamp', 'direction', 'outcome', 'balance_after']:
            assert k in lb

    def test_stub_payload_ts_is_recent(self):
        import time
        broadcaster = WSBroadcaster(pool=None)
        payload = broadcaster._stub_payload()
        assert abs(payload['ts'] - time.time()) < 2.0

    def test_build_payload_no_db(self):
        broadcaster = WSBroadcaster(pool=None)
        payload = run_async(broadcaster._build_payload())
        assert payload['bot_state'] == 'BOOT'

    def test_client_count_starts_zero(self):
        broadcaster = WSBroadcaster(pool=None)
        assert broadcaster.client_count == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
