import nbformat
import os

def update():
    nb_path = 'notebooks/04_backtest_results.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 1. Run Backtest' in cell.source:
            code_cell = nb.cells[i+1]
            
            new_code = """import os
import numpy as np
import pandas as pd

data_dir = os.path.abspath(os.path.join('..', 'data', 'processed'))

# Load the real Walk-Forward Validation out-of-sample predictions and labels
oos_probs_path = os.path.join(data_dir, 'oos_probs.npy')
oos_y_path = os.path.join(data_dir, 'oos_y.npy')
timestamps_path = os.path.join(data_dir, 'ts_180d.npy')

if not os.path.exists(oos_probs_path) or not os.path.exists(oos_y_path):
    raise FileNotFoundError("OOS data not found! Please go back to Notebook 03, run the final cell to save 'oos_probs.npy' and 'oos_y.npy', then try again.")

probs = np.load(oos_probs_path)
actuals = np.load(oos_y_path)
timestamps_full = np.load(timestamps_path)

# Ensure the lengths match (WF Validation might drop the first few days of training data)
n_bars = len(probs)
timestamps = timestamps_full[-n_bars:]

# Suppress verbose logging to prevent notebook freezing
import src.sizing.kelly_sizer
import src.sizing.compounding_engine
import src.signal.signal_generator
dummy_logger = type('DummyLogger', (), {'debug': lambda *a, **kw: None, 'info': lambda *a, **kw: None, 'warning': lambda *a, **kw: None, 'error': lambda *a, **kw: None})()
src.sizing.kelly_sizer.logger = dummy_logger
src.sizing.compounding_engine.logger = dummy_logger
src.signal.signal_generator.logger = dummy_logger

print(f"Loaded {n_bars} bars of historical out-of-sample backtest data.")
print("Running Backtester...")

# Run Backtest Engine
bt = Backtester(initial_capital=10.0, signal_threshold=0.55, high_conf_threshold=0.70)
result = bt.run(probabilities=probs, timestamps=timestamps, actuals=actuals)

# Generate Report
report = ReportGenerator().generate(result)

print(f"Backtest complete. Final capital: ${result.final_capital:.2f}")
"""
            code_cell.source = new_code

        if cell.cell_type == 'markdown' and '## 5. Leakage Check' in cell.source:
            code_cell = nb.cells[i+1]
            
            new_code = """import os
import numpy as np
from src.backtest.leakage_detector import LeakageDetector

data_dir = os.path.abspath(os.path.join('..', 'data', 'processed'))
X = np.load(os.path.join(data_dir, 'X_180d.npy'))

# Limit X to the exact OOS period we just backtested
X_oos = X[-n_bars:]

ld = LeakageDetector()
train_idx = np.arange(n_bars * 3 // 4)
val_idx = np.arange(n_bars * 3 // 4, n_bars)

# Run Leakage Detector on REAL data instead of random noise
leakage = ld.run_all(X_oos, actuals, timestamps, train_idx, val_idx)

print('=== LEAKAGE REPORT ===')
for c in leakage.checks:
    status = 'PASS' if c.passed else 'FAIL'
    print(f'  [{status}] {c.name}: {c.details}')
"""
            code_cell.source = new_code

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Updated 04_backtest_results.ipynb to use REAL OOS data")

if __name__ == '__main__':
    update()
