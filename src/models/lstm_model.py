"""
LSTM Model — Temporal ensemble member (weight: 0.35).

Captures sequential dependencies in feature time-series that
tree-based models cannot model (regime transitions, momentum
carry-over across bars).

Architecture (PRD 6.3.2):
  Input:  (batch, 30, n_features)  — last 30 bars × 150 features
  LSTM-1: 128 units, Dropout(0.3)
  LSTM-2: 64 units,  Dropout(0.3)
  Dense:  32 units,  ReLU
  Output: 1 unit,    Sigmoid → P(UP)

Data Leakage Prevention:
  - Sequence construction uses ONLY past bars (no lookahead)
  - Normalisation statistics (mean/std) computed on training set only
  - Same temporal split contract as tree models
"""

import time
import json
from typing import Optional
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Sequence length: last N bars per sample
SEQ_LENGTH = 30

# Try PyTorch first (preferred), then TensorFlow, then stub
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    if not HAS_TORCH:
        import tensorflow as tf
        from tensorflow import keras
        HAS_TF = True
    else:
        HAS_TF = False
except ImportError:
    HAS_TF = False


class LSTMModel:
    """
    LSTM binary classifier wrapper with consistent ensemble interface.

    Provides:
      - fit(X_seq, y) — train on 3D sequence data
      - predict_proba(X_seq) — returns P(UP) for each sample
      - save/load — serialisation

    When neither PyTorch nor TensorFlow is available, falls back to
    a simple logistic stub (for testing / environments without GPU).
    """

    def __init__(self, n_features: int = 150, seq_length: int = SEQ_LENGTH):
        self._n_features = n_features
        self._seq_length = seq_length
        self._model = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._train_metadata: dict = {}
        self._backend = 'none'

        if HAS_TORCH:
            self._backend = 'torch'
        elif HAS_TF:
            self._backend = 'tensorflow'
        else:
            self._backend = 'stub'
            logger.warning("lstm_no_backend", msg="Using logistic stub — install torch or tensorflow")

    @staticmethod
    def build_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int = SEQ_LENGTH,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build 3D sequence arrays from 2D feature matrix.

        Data Leakage Prevention:
          Each sequence X_seq[i] contains rows [i : i+seq_length].
          The label y_seq[i] is the label at position i+seq_length-1.
          NO future data is included in any sequence.

        Args:
            X: 2D array (n_samples, n_features), chronologically ordered.
            y: 1D array (n_samples,) labels.
            seq_length: Number of time steps per sequence.

        Returns:
            X_seq: 3D array (n_sequences, seq_length, n_features).
            y_seq: 1D array (n_sequences,).
        """
        n = len(X) - seq_length + 1
        if n <= 0:
            return np.array([]).reshape(0, seq_length, X.shape[1]), np.array([])

        X_seq = np.empty((n, seq_length, X.shape[1]), dtype=np.float32)
        y_seq = np.empty(n, dtype=np.float32)

        for i in range(n):
            X_seq[i] = X[i:i + seq_length]
            y_seq[i] = y[i + seq_length - 1]

        return X_seq, y_seq

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 64,
    ) -> dict:
        """
        Train on 3D sequence data with early stopping.

        Data Leakage Prevention:
          - Normalisation (mean/std) computed on X_train ONLY
          - X_val normalised using X_train's statistics
          - Sequences must be pre-built by caller using build_sequences()

        Args:
            X_train: 3D (n_train, seq_len, n_features).
            y_train: 1D (n_train,).
            X_val: Optional 3D validation data.
            y_val: Optional 1D validation labels.
            epochs: Max training epochs.
            patience: Early stopping patience.
            batch_size: Mini-batch size.

        Returns:
            Training metadata dict.
        """
        t0 = time.perf_counter()

        # Compute normalisation on training set only (leakage prevention)
        self._scaler_mean = X_train.mean(axis=(0, 1))
        self._scaler_std = X_train.std(axis=(0, 1))
        self._scaler_std = np.where(self._scaler_std < 1e-10, 1.0, self._scaler_std)

        X_train_norm = (X_train - self._scaler_mean) / self._scaler_std
        X_val_norm = None
        if X_val is not None:
            X_val_norm = (X_val - self._scaler_mean) / self._scaler_std

        self._n_features = X_train.shape[2]

        if self._backend == 'torch':
            self._fit_torch(X_train_norm, y_train, X_val_norm, y_val, epochs, patience, batch_size)
        elif self._backend == 'tensorflow':
            self._fit_tf(X_train_norm, y_train, X_val_norm, y_val, epochs, patience, batch_size)
        else:
            self._fit_stub(X_train_norm, y_train)

        elapsed = time.perf_counter() - t0
        self._train_metadata = {
            'model_type': 'lstm',
            'backend': self._backend,
            'n_samples_train': len(y_train),
            'n_samples_val': len(y_val) if y_val is not None else 0,
            'n_features': self._n_features,
            'seq_length': self._seq_length,
            'train_time_sec': round(elapsed, 2),
        }
        logger.info("lstm_trained", backend=self._backend, samples=len(y_train), elapsed_sec=round(elapsed, 2))
        return self._train_metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(UP) for 3D sequence input. Shape: (n_samples,)."""
        if self._scaler_mean is not None:
            X = (X - self._scaler_mean) / self._scaler_std

        if self._backend == 'torch':
            return self._predict_torch(X)
        elif self._backend == 'tensorflow':
            return self._predict_tf(X)
        else:
            return self._predict_stub(X)

    # ────────────────────── PyTorch backend ──────────────────────────

    def _fit_torch(self, X, y, X_val, y_val, epochs, patience, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _TorchLSTM(X.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        best_loss = float('inf')
        wait = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X_t))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
                    val_pred = model(val_t).squeeze(-1)
                    val_loss = criterion(val_pred, torch.tensor(y_val, dtype=torch.float32, device=device)).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

        if best_state:
            model.load_state_dict(best_state)
        self._model = model.cpu().eval()

    def _predict_torch(self, X):
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return self._model(X_t).squeeze(-1).numpy()

    # ────────────────────── TensorFlow backend ──────────────────────

    def _fit_tf(self, X, y, X_val, y_val, epochs, patience, batch_size):
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        callbacks = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)]
        val_data = (X_val, y_val) if X_val is not None else None
        model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=val_data,
                  callbacks=callbacks, verbose=0)
        self._model = model

    def _predict_tf(self, X):
        return self._model.predict(X, verbose=0).flatten()

    # ────────────────────── Stub backend ────────────────────────────

    def _fit_stub(self, X, y):
        """Simple logistic regression on flattened last timestep."""
        self._stub_weights = np.zeros(X.shape[2])
        self._stub_bias = 0.0
        last_step = X[:, -1, :]
        mean_pos = last_step[y == 1].mean(axis=0) if np.sum(y == 1) > 0 else np.zeros(X.shape[2])
        mean_neg = last_step[y == 0].mean(axis=0) if np.sum(y == 0) > 0 else np.zeros(X.shape[2])
        diff = mean_pos - mean_neg
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            self._stub_weights = diff / norm
        self._stub_bias = -np.dot(self._stub_weights, (mean_pos + mean_neg) / 2)
        self._model = 'stub'

    def _predict_stub(self, X):
        last_step = X[:, -1, :]
        logits = last_step @ self._stub_weights + self._stub_bias
        return 1.0 / (1.0 + np.exp(-logits))

    # ────────────────────── Serialisation ───────────────────────────

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self._backend == 'torch' and self._model is not None:
            torch.save(self._model.state_dict(), str(path / 'lstm_model.pt'))
        elif self._backend == 'tensorflow' and self._model is not None:
            self._model.save(str(path / 'lstm_model.keras'))
        np.savez(path / 'lstm_scaler.npz',
                 mean=self._scaler_mean if self._scaler_mean is not None else np.array([]),
                 std=self._scaler_std if self._scaler_std is not None else np.array([]))
        with open(path / 'lstm_metadata.json', 'w') as f:
            json.dump(self._train_metadata, f, indent=2)
        logger.info("lstm_saved", path=str(path))

    @classmethod
    def load(cls, directory: str, n_features: int = 150) -> 'LSTMModel':
        path = Path(directory)
        instance = cls(n_features=n_features)
        scaler = np.load(path / 'lstm_scaler.npz')
        instance._scaler_mean = scaler['mean']
        instance._scaler_std = scaler['std']
        if instance._backend == 'torch':
            model = _TorchLSTM(n_features)
            model.load_state_dict(torch.load(str(path / 'lstm_model.pt'), weights_only=True))
            instance._model = model.eval()
        elif instance._backend == 'tensorflow':
            instance._model = keras.models.load_model(str(path / 'lstm_model.keras'))
        meta_path = path / 'lstm_metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                instance._train_metadata = json.load(f)
        logger.info("lstm_loaded", path=str(path))
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._model is not None


# ─── PyTorch LSTM Module ─────────────────────────────────────────────

if HAS_TORCH:
    class _TorchLSTM(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm1 = nn.LSTM(n_features, 128, batch_first=True)
            self.drop1 = nn.Dropout(0.3)
            self.lstm2 = nn.LSTM(128, 64, batch_first=True)
            self.drop2 = nn.Dropout(0.3)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.drop1(x)
            x, _ = self.lstm2(x)
            x = self.drop2(x[:, -1, :])  # last timestep
            x = torch.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))
