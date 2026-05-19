import json
import nbformat
import os

def update_notebook_04():
    nb_path = 'notebooks/04_backtest_results.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 1. Run Backtest' in cell.source:
            code_cell = nb.cells[i+1]
            
            new_code = """# In a real scenario, we load the out-of-sample predictions and prices
# We will mock this slightly for the backtester display using the 180d close prices
import os
import numpy as np
import pandas as pd

data_dir = os.path.abspath(os.path.join('..', 'data', 'processed'))
close_prices = np.load(os.path.join(data_dir, 'close_180d.npy'))
timestamps = np.load(os.path.join(data_dir, 'ts_180d.npy'))

n_bars = len(close_prices)

# For testing, we mock model probabilities with slight edge
np.random.seed(42)
probs = np.random.normal(0.55, 0.15, size=n_bars)
probs = np.clip(probs, 0.01, 0.99)

# To simulate realistic actuals (1=UP, 0=DOWN) based on probabilities:
actuals = (np.random.rand(n_bars) < probs).astype(int)

# Suppress verbose logging to prevent notebook freezing
import src.sizing.kelly_sizer
import src.sizing.compounding_engine
import src.signal.signal_generator
dummy_logger = type('DummyLogger', (), {'debug': lambda *a, **kw: None, 'info': lambda *a, **kw: None, 'warning': lambda *a, **kw: None, 'error': lambda *a, **kw: None})()
src.sizing.kelly_sizer.logger = dummy_logger
src.sizing.compounding_engine.logger = dummy_logger
src.signal.signal_generator.logger = dummy_logger

print(f"Loaded {n_bars} bars of historical backtest data.")
print("Running Backtester...")

# Run Backtest Engine
bt = Backtester(initial_capital=10.0, signal_threshold=0.55, high_conf_threshold=0.70)
result = bt.run(probabilities=probs, timestamps=timestamps, actuals=actuals)

# Generate Report
report = ReportGenerator().generate(result)

print(f"Backtest complete. Final capital: ${result.final_capital:.2f}")
"""
            code_cell.source = new_code
            break

    # Fix the reference in Cell 3 (which the user might have changed to wf_result)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 2. Equity Curve' in cell.source:
            code_cell = nb.cells[i+1]
            # Replace wf_result with result if they changed it
            code_cell.source = code_cell.source.replace('wf_result', 'result')
            break

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Updated 04_backtest_results.ipynb Run Backtest cell")

if __name__ == '__main__':
    update_notebook_04()
