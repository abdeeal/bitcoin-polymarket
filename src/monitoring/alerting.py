"""
Alerting — Telegram + Email notifications for 10 alert rules.

PRD 14.1 Alert Rules:
  1. Feed Down         — No candle for >2 min          (Telegram + Email)
  2. Cooling Down Start — 2 consecutive losses         (Telegram)
  3. Cooling Down End   — 30-min pause complete         (Telegram)
  4. Hard Stop          — Balance < $5.00               (Telegram + Email)
  5. Bracket Upgrade    — Balance crossed ceiling       (Telegram)
  6. Bracket Downgrade  — Balance dropped below floor   (Telegram)
  7. Model Drift        — PSI > 0.25 on top 10 features (Email)
  8. Win Rate Drop      — 7-day WR < 50%               (Email)
  9. API Error Rate     — >5% failed calls in 1h       (Telegram)
 10. Feature Latency    — P95 > 100ms                  (Telegram)

Architecture:
  - AlertManager holds all rules and dispatches via channels.
  - Channels: TelegramChannel, EmailChannel, LogChannel (fallback).
  - Rate-limited: same alert type suppressed for 5 minutes.
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import structlog

logger = structlog.get_logger(__name__)

RATE_LIMIT_SECONDS = 300  # 5 minutes between duplicate alerts


class AlertSeverity(Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class AlertChannel(Enum):
    TELEGRAM = "TELEGRAM"
    EMAIL    = "EMAIL"
    LOG      = "LOG"


@dataclass
class Alert:
    """A single alert event."""
    name: str
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    channels: list[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'message': self.message,
            'severity': self.severity.value,
            'channels': [c.value for c in self.channels],
            'timestamp': self.timestamp,
        }


# ── Channel Implementations ──────────────────────────────────────────────────

class BaseChannel(ABC):
    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert. Returns True if delivered."""
        ...


class LogChannel(BaseChannel):
    """Fallback: logs the alert via structlog."""
    async def send(self, alert: Alert) -> bool:
        logger.info("alert_fired", name=alert.name, msg=alert.message, severity=alert.severity.value)
        return True


class TelegramChannel(BaseChannel):
    """
    Send alerts to Telegram via Bot API.

    Requires env vars:
      TELEGRAM_BOT_TOKEN  — bot token from @BotFather
      TELEGRAM_CHAT_ID    — chat/group ID to send to
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    async def send(self, alert: Alert) -> bool:
        if not self._token or not self._chat_id:
            logger.warning("telegram_not_configured", alert=alert.name)
            return False

        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }
        emoji = severity_emoji.get(alert.severity, "📢")
        text = f"{emoji} *{alert.name}*\n{alert.message}"

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                }) as resp:
                    ok = resp.status == 200
                    if not ok:
                        body = await resp.text()
                        logger.error("telegram_send_failed", status=resp.status, body=body)
                    return ok
        except ImportError:
            logger.warning("aiohttp_not_installed", alert=alert.name)
            return False
        except Exception as e:
            logger.error("telegram_error", error=str(e))
            return False


class EmailChannel(BaseChannel):
    """
    Send alerts via SMTP (async wrapper around smtplib).

    Requires env vars:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO
    """

    def __init__(self):
        self._host = os.getenv("SMTP_HOST", "")
        self._port = int(os.getenv("SMTP_PORT", "587"))
        self._user = os.getenv("SMTP_USER", "")
        self._pass = os.getenv("SMTP_PASS", "")
        self._to = os.getenv("ALERT_EMAIL_TO", "")

    async def send(self, alert: Alert) -> bool:
        if not self._host or not self._to:
            logger.warning("email_not_configured", alert=alert.name)
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(f"{alert.severity.value}: {alert.message}")
            msg['Subject'] = f"[BTC Bot] {alert.name}"
            msg['From'] = self._user
            msg['To'] = self._to

            # Run in thread pool (smtplib is blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp, msg)
            return True
        except Exception as e:
            logger.error("email_send_failed", error=str(e))
            return False

    def _send_smtp(self, msg):
        import smtplib
        with smtplib.SMTP(self._host, self._port) as server:
            server.starttls()
            server.login(self._user, self._pass)
            server.send_message(msg)


# ── Alert Manager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Central alert dispatcher with rate limiting.

    Dispatches alerts to the appropriate channels based on alert config.
    Rate-limits: same alert name suppressed for 5 minutes.

    Usage:
        mgr = AlertManager()
        await mgr.fire(Alert("Feed Down", "No candle for >2 min",
                             channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL]))
    """

    def __init__(
        self,
        telegram: Optional[TelegramChannel] = None,
        email: Optional[EmailChannel] = None,
    ):
        self._channels: dict[AlertChannel, BaseChannel] = {
            AlertChannel.LOG: LogChannel(),
        }
        if telegram:
            self._channels[AlertChannel.TELEGRAM] = telegram
        if email:
            self._channels[AlertChannel.EMAIL] = email

        self._last_fired: dict[str, float] = {}
        self._history: list[Alert] = []

    async def fire(self, alert: Alert) -> bool:
        """
        Dispatch alert to all configured channels.
        Rate-limited: same alert name suppressed for RATE_LIMIT_SECONDS.

        Returns True if at least one channel delivered.
        """
        now = time.time()
        last = self._last_fired.get(alert.name, 0)
        if now - last < RATE_LIMIT_SECONDS:
            logger.debug("alert_rate_limited", name=alert.name,
                         remaining=round(RATE_LIMIT_SECONDS - (now - last)))
            return False

        self._last_fired[alert.name] = now
        self._history.append(alert)

        delivered = False
        for ch_type in alert.channels:
            channel = self._channels.get(ch_type)
            if channel is None:
                # Fall back to LOG
                channel = self._channels[AlertChannel.LOG]
            ok = await channel.send(alert)
            delivered = delivered or ok

        # Always log regardless of channel delivery
        if AlertChannel.LOG not in alert.channels:
            await self._channels[AlertChannel.LOG].send(alert)

        return delivered

    @property
    def history(self) -> list[Alert]:
        return list(self._history)

    def clear_rate_limits(self) -> None:
        """Reset all rate limits (for testing)."""
        self._last_fired.clear()


# ── Pre-built Alert Factory ──────────────────────────────────────────────────

def alert_feed_down(seconds_since: float) -> Alert:
    """Alert 1: No candle received for >2 minutes."""
    return Alert(
        name="Feed Down",
        message=f"No candle received for {seconds_since:.0f}s (threshold: 120s)",
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
    )


def alert_cooling_down_start(consecutive_losses: int, balance: float) -> Alert:
    """Alert 2: Cooling down triggered by consecutive losses."""
    return Alert(
        name="Cooling Down Start",
        message=f"Cooling down triggered: {consecutive_losses} consecutive losses. Balance: ${balance:.2f}",
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.TELEGRAM],
    )


def alert_cooling_down_end(duration_sec: float) -> Alert:
    """Alert 3: Cooling down period complete."""
    return Alert(
        name="Cooling Down End",
        message=f"Cooling down ended after {duration_sec:.0f}s. Resuming active trading.",
        severity=AlertSeverity.INFO,
        channels=[AlertChannel.TELEGRAM],
    )


def alert_hard_stop(balance: float) -> Alert:
    """Alert 4: Balance below hard stop threshold."""
    return Alert(
        name="Hard Stop",
        message=f"Balance ${balance:.2f} below $5.00 hard stop threshold.",
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
    )


def alert_bracket_upgrade(from_bracket: int, to_bracket: int, balance: float) -> Alert:
    """Alert 5: Balance crossed bracket ceiling (upgrade)."""
    return Alert(
        name="Bracket Upgrade",
        message=f"Upgraded from bracket {from_bracket} → {to_bracket}. Balance: ${balance:.2f}",
        severity=AlertSeverity.INFO,
        channels=[AlertChannel.TELEGRAM],
    )


def alert_bracket_downgrade(from_bracket: int, to_bracket: int, balance: float) -> Alert:
    """Alert 6: Balance dropped below bracket floor (downgrade)."""
    return Alert(
        name="Bracket Downgrade",
        message=f"Downgraded from bracket {from_bracket} → {to_bracket}. Balance: ${balance:.2f}",
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.TELEGRAM],
    )


def alert_model_drift(feature_name: str, psi_value: float) -> Alert:
    """Alert 7: PSI > 0.25 on a top feature → model drift."""
    return Alert(
        name="Model Drift",
        message=f"Feature '{feature_name}' PSI = {psi_value:.3f} (threshold: 0.25). Retrain recommended.",
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.EMAIL],
    )


def alert_win_rate_drop(win_rate: float, period_days: int = 7) -> Alert:
    """Alert 8: 7-day win rate dropped below 50%."""
    return Alert(
        name="Win Rate Drop",
        message=f"{period_days}-day win rate = {win_rate:.1%} (below 50% threshold).",
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.EMAIL],
    )


def alert_api_error_rate(error_pct: float, window_minutes: int = 60) -> Alert:
    """Alert 9: API error rate exceeded 5% in 1 hour."""
    return Alert(
        name="API Error Rate",
        message=f"API error rate = {error_pct:.1f}% in last {window_minutes} min (threshold: 5%).",
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.TELEGRAM],
    )


def alert_feature_latency(p95_ms: float) -> Alert:
    """Alert 10: Feature computation P95 latency exceeded 100ms."""
    return Alert(
        name="Feature Latency",
        message=f"Feature P95 latency = {p95_ms:.1f}ms (budget: 100ms).",
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.TELEGRAM],
    )
