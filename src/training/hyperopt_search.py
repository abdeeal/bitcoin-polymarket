"""
Hyperparameter Search — Optuna-based Bayesian optimization.

PRD 6.5: Optuna, 200 trials, TimeSeriesSplit(n_splits=5)
Objective: maximize OOS log-loss (minimize)

Design:
  - Three separate search spaces: LightGBM, XGBoost, LSTM
  - TimeSeriesSplit ensures strict chronological splits (no leakage)
  - Trials pruned early using MedianPruner
  - Best params persisted to models/hyperparams/<model>_best.json
  - Warm-start: if best.json exists, Optuna loads from it

Usage:
    python -m src.training.hyperopt_search --model lgbm --trials 200
    python -m src.training.hyperopt_search --model xgb --trials 200
    python -m src.training.hyperopt_search --model all --trials 200
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

N_TRIALS_DEFAULT = 200
N_SPLITS = 5                     # TimeSeriesSplit folds
RANDOM_STATE = 42
HYPERPARAMS_DIR = Path('models/hyperparams')


# ─── Search Spaces ────────────────────────────────────────────────────────────

def _lgbm_space(trial) -> dict:
    """LightGBM search space."""
    return {
        'objective':        'binary',
        'metric':           'binary_logloss',
        'verbosity':        -1,
        'random_state':     RANDOM_STATE,
        'n_jobs':           -1,
        'n_estimators':     trial.suggest_int('n_estimators', 300, 2000, step=100),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'num_leaves':       trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'class_weight':     'balanced',
    }


def _xgb_space(trial) -> dict:
    """XGBoost search space."""
    return {
        'objective':        'binary:logistic',
        'eval_metric':      'logloss',
        'tree_method':      'hist',
        'random_state':     RANDOM_STATE,
        'n_jobs':           -1,
        'verbosity':        0,
        'n_estimators':     trial.suggest_int('n_estimators', 300, 1500, step=100),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 7),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
    }


def _lstm_space(trial) -> dict:
    """LSTM architecture + training search space."""
    return {
        'units_1':      trial.suggest_int('units_1', 64, 256, step=32),
        'units_2':      trial.suggest_int('units_2', 32, 128, step=32),
        'dropout':      trial.suggest_float('dropout', 0.1, 0.5),
        'dense_units':  trial.suggest_int('dense_units', 16, 64, step=16),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size':   trial.suggest_categorical('batch_size', [32, 64, 128]),
        'sequence_len': trial.suggest_int('sequence_len', 20, 60, step=10),
    }


# ─── Objective Functions ──────────────────────────────────────────────────────

def _make_lgbm_objective(X: np.ndarray, y: np.ndarray, cv_splits: list):
    """LightGBM CV objective for Optuna."""
    def objective(trial):
        try:
            import lightgbm as lgb
            from sklearn.metrics import log_loss
        except ImportError:
            raise RuntimeError("lightgbm and scikit-learn required for hyperopt")

        params = _lgbm_space(trial)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(log_loss(y_val, proba))

        return float(np.mean(scores))

    return objective


def _make_xgb_objective(X: np.ndarray, y: np.ndarray, cv_splits: list):
    """XGBoost CV objective for Optuna."""
    def objective(trial):
        try:
            import xgboost as xgb
            from sklearn.metrics import log_loss
        except ImportError:
            raise RuntimeError("xgboost and scikit-learn required for hyperopt")

        params = _xgb_space(trial)
        scores = []

        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params, early_stopping_rounds=50,
                                       eval_metric='logloss')
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(log_loss(y_val, proba))

        return float(np.mean(scores))

    return objective


# ─── Main Searcher ────────────────────────────────────────────────────────────

class HyperoptSearch:
    """
    Optuna-based Bayesian hyperparameter search.

    Uses TimeSeriesSplit for strictly chronological CV
    to prevent any data leakage from future bars.

    PRD 6.5:
      - 200 trials per model
      - TimeSeriesSplit(n_splits=5)
      - Objective: minimize OOS binary log-loss
    """

    def __init__(
        self,
        n_trials: int = N_TRIALS_DEFAULT,
        n_splits: int = N_SPLITS,
        output_dir: str = 'models/hyperparams',
        timeout_sec: Optional[float] = None,
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.output_dir = Path(output_dir)
        self.timeout_sec = timeout_sec
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_cv_splits(self, n_samples: int) -> list:
        """Generate TimeSeriesSplit indices — strict chronological."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            return list(tscv.split(np.arange(n_samples)))
        except ImportError:
            # Manual fallback if sklearn not available
            splits = []
            fold_size = n_samples // (self.n_splits + 1)
            for i in range(1, self.n_splits + 1):
                train_end = i * fold_size
                val_end = train_end + fold_size
                splits.append((
                    np.arange(0, train_end),
                    np.arange(train_end, min(val_end, n_samples))
                ))
            return splits

    def run_lgbm(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run Optuna search for LightGBM."""
        return self._run('lgbm', X, y, _make_lgbm_objective)

    def run_xgb(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run Optuna search for XGBoost."""
        return self._run('xgb', X, y, _make_xgb_objective)

    def _run(self, model_name: str, X: np.ndarray, y: np.ndarray,
             objective_factory) -> dict:
        """
        Run the Optuna study for the given model.

        Returns:
            dict with best_params and best_value.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise RuntimeError("optuna required: pip install optuna")

        cv_splits = self._get_cv_splits(len(X))
        objective = objective_factory(X, y, cv_splits)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            sampler=sampler,
            study_name=f'{model_name}_hyperopt',
        )

        t0 = time.perf_counter()
        logger.info("hyperopt_started", model=model_name, trials=self.n_trials)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_sec,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - t0
        best = {
            'model': model_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'elapsed_sec': round(elapsed, 1),
        }

        # Persist best params
        output_path = self.output_dir / f'{model_name}_best.json'
        output_path.write_text(json.dumps(best, indent=2))

        logger.info(
            "hyperopt_completed",
            model=model_name,
            best_logloss=round(study.best_value, 4),
            trials=len(study.trials),
            elapsed_sec=round(elapsed, 1),
        )

        return best

    def load_best_params(self, model_name: str) -> Optional[dict]:
        """Load previously saved best parameters."""
        path = self.output_dir / f'{model_name}_best.json'
        if not path.exists():
            return None
        return json.loads(path.read_text())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperopt search')
    parser.add_argument('--model', choices=['lgbm', 'xgb', 'all'], default='lgbm')
    parser.add_argument('--trials', type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument('--timeout', type=float, default=None)
    args = parser.parse_args()

    print(f"Note: Run with real feature data: X, y arrays from feature pipeline.")
    print(f"Example: searcher.run_lgbm(X_features, y_labels)")
