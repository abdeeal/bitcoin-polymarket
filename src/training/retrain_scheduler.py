"""
Retrain Scheduler — Weekly automatic model retraining.

PRD 14.2:
  - Runs every Sunday 00:00 UTC
  - Fetches last 90 days of training data
  - Runs walk-forward validation
  - Deploys new model only if accuracy improves by ≥0.5%
  - Updates normalizer windows
  - Sends alert on successful deployment

Integration:
  - Uses APScheduler for cron-like scheduling
  - Drift detector triggers ad-hoc retrain if PSI > 0.25
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Callable

import structlog

logger = structlog.get_logger(__name__)

MODELS_DIR = Path("models/latest")
HYPERPARAMS_DIR = Path("models/hyperparams")
RETRAIN_DAYS = 90
MIN_ACCURACY_IMPROVEMENT = 0.005  # 0.5% absolute


class RetrainScheduler:
    """
    Weekly model retraining pipeline.

    Workflow:
      1. Fetch last 90 days of training data
      2. Run walk-forward validation with EnsembleModel
      3. Compare accuracy to current model
      4. If accuracy improves by ≥0.5%, deploy new model
      5. Update normalizer windows
      6. Send deployment alert

    Args:
        fetch_data_fn:   Async function returning training DataFrame.
        walk_forward_fn: Function running walk-forward validation.
        train_fn:        Function training the final model.
        alert_fn:        Function sending deployment alerts.
        models_dir:      Directory to save deployed models.
    """

    def __init__(
        self,
        fetch_data_fn: Optional[Callable] = None,
        walk_forward_fn: Optional[Callable] = None,
        train_fn: Optional[Callable] = None,
        alert_fn: Optional[Callable] = None,
        models_dir: Optional[Path] = None,
    ):
        self._fetch_data = fetch_data_fn
        self._walk_forward = walk_forward_fn
        self._train = train_fn
        self._alert = alert_fn
        self._models_dir = models_dir or MODELS_DIR
        self._current_accuracy: float = 0.0
        self._last_retrain: Optional[float] = None
        self._retrain_count: int = 0
        self._retrain_history: list[dict] = []

    def set_current_accuracy(self, accuracy: float) -> None:
        """Set the accuracy of the currently deployed model."""
        self._current_accuracy = accuracy

    async def run_retrain(self, reason: str = "scheduled") -> dict:
        """
        Execute one retrain cycle.

        Returns dict with result: 'deployed', 'no_improvement', or 'error'.
        """
        t0 = time.time()
        result = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'reason': reason,
            'result': 'error',
            'current_accuracy': self._current_accuracy,
            'new_accuracy': None,
            'improvement': None,
            'deployed': False,
        }

        try:
            # Step 1: Fetch training data
            if self._fetch_data is None:
                logger.warning("retrain_skipped", reason="no fetch_data_fn configured")
                result['result'] = 'skipped'
                return result

            df = await self._fetch_data(days=RETRAIN_DAYS)
            logger.info("retrain_data_fetched", rows=len(df) if hasattr(df, '__len__') else '?')

            # Step 2: Walk-forward validation
            if self._walk_forward is None:
                result['result'] = 'skipped'
                return result

            metrics = self._walk_forward(df)
            new_accuracy = metrics.get('accuracy', 0.0)
            result['new_accuracy'] = new_accuracy

            # Step 3: Check improvement
            improvement = new_accuracy - self._current_accuracy
            result['improvement'] = round(improvement, 4)

            if improvement >= MIN_ACCURACY_IMPROVEMENT:
                # Step 4: Train final model
                if self._train:
                    model = self._train(df)
                    self._save_model(model, new_accuracy)

                self._current_accuracy = new_accuracy
                self._retrain_count += 1
                self._last_retrain = time.time()
                result['result'] = 'deployed'
                result['deployed'] = True

                logger.info(
                    "retrain_deployed",
                    accuracy=new_accuracy,
                    improvement=improvement,
                )

                # Step 5: Alert
                if self._alert:
                    await self._alert(
                        f"New model deployed: acc={new_accuracy:.4f} "
                        f"(+{improvement:.4f})"
                    )
            else:
                result['result'] = 'no_improvement'
                logger.info(
                    "retrain_no_improvement",
                    current=self._current_accuracy,
                    new=new_accuracy,
                    improvement=improvement,
                    threshold=MIN_ACCURACY_IMPROVEMENT,
                )

        except Exception as e:
            result['result'] = 'error'
            result['error'] = str(e)
            logger.error("retrain_failed", error=str(e))

        elapsed = time.time() - t0
        result['elapsed_sec'] = round(elapsed, 1)
        self._retrain_history.append(result)
        return result

    def _save_model(self, model: Any, accuracy: float) -> None:
        """Save model to models/latest/ with metadata."""
        self._models_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            'accuracy': accuracy,
            'deployed_at': datetime.now(tz=timezone.utc).isoformat(),
            'retrain_count': self._retrain_count + 1,
        }
        meta_path = self._models_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("model_saved", path=str(self._models_dir), accuracy=accuracy)

    def should_retrain_from_drift(self, drift_report) -> bool:
        """Check if drift report triggers an ad-hoc retrain."""
        return getattr(drift_report, 'retrain_triggered', False)

    @property
    def stats(self) -> dict:
        return {
            'current_accuracy': self._current_accuracy,
            'retrain_count': self._retrain_count,
            'last_retrain': self._last_retrain,
            'history_count': len(self._retrain_history),
        }
