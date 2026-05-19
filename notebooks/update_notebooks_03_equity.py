import json
import nbformat
import os

def update_notebook_03():
    nb_path = 'notebooks/03_walk_forward_results.ipynb'
    if not os.path.exists(nb_path):
        print(f"File not found: {nb_path}")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # find the cell after "## 4. Equity Curve Simulation"
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 4. Equity Curve Simulation' in cell.source:
            # The next cell is the simulation code cell
            code_cell = nb.cells[i+1]
            
            new_code = """from src.sizing.compounding_engine import CompoundingEngine
from src.signal.signal_generator import Signal, SignalGenerator, BotState
import numpy as np
import matplotlib.pyplot as plt

# Collect out-of-sample predictions and targets from all folds
all_val_probs = []
all_y_val = []

if hasattr(wf_result, 'folds') and len(wf_result.folds) > 0 and hasattr(wf_result.folds[0], 'val_probs') and wf_result.folds[0].val_probs is not None:
    for fold in wf_result.folds:
        all_val_probs.extend(fold.val_probs)
        all_y_val.extend(fold.y_val)
else:
    print("Warning: val_probs not found in fold results. Please re-run walk-forward validation.")
    # Fallback for testing
    np.random.seed(42)
    all_y_val = np.random.randint(0, 2, 500)
    all_val_probs = np.random.rand(500)

all_val_probs = np.array(all_val_probs)
all_y_val = np.array(all_y_val)

# Suppress verbose logging to prevent notebook freezing during 23k iterations
import src.sizing.kelly_sizer
import src.sizing.compounding_engine
dummy_logger = type('DummyLogger', (), {'debug': lambda *a, **kw: None, 'info': lambda *a, **kw: None, 'warning': lambda *a, **kw: None, 'error': lambda *a, **kw: None})()
src.sizing.kelly_sizer.logger = dummy_logger
src.sizing.compounding_engine.logger = dummy_logger

# Use CompoundingEngine to simulate the equity curve
ce = CompoundingEngine(10.0)
gen = SignalGenerator()
balance_curve = [10.0]
cooling_periods = []
consecutive_losses = 0
cooling_down_until = -1

for i, (prob, y_true) in enumerate(zip(all_val_probs, all_y_val)):
    if i < cooling_down_until:
        balance_curve.append(ce.balance)
        continue
    
    # Generate signal based on PRD thresholds via SignalGenerator
    sig_result = gen.generate_signal(prob, filters_ok=True, bot_state=BotState.ACTIVE)
    signal = sig_result.signal
        
    bet = ce.resolve_bet_size(signal, prob=prob)
    
    if bet == 0:
        balance_curve.append(ce.balance)
        continue
        
    won = (prob >= 0.5) == y_true
    payout = bet * 1.90 if won else 0.0
    ce.on_bet_resolved(won=won, bet_amount=bet, payout=payout)
    balance_curve.append(ce.balance)
    
    if won:
        consecutive_losses = 0
    else:
        consecutive_losses += 1
        if consecutive_losses >= 2:
            cooling_down_until = i + 6  # 30min = 6 bars
            cooling_periods.append((i, i + 6))
            consecutive_losses = 0

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(balance_curve, linewidth=1.5, color='steelblue', label='Balance')
if cooling_periods:
    for start, end in cooling_periods:
        ax.axvspan(start, end, alpha=0.15, color='red', label='Cooling Down' if start == cooling_periods[0][0] else '')
ax.axhline(10.0, color='gray', linestyle='--', linewidth=0.8, label='Start $10')

n_bets = len(all_y_val)
win_rate = np.mean((all_val_probs >= 0.5) == all_y_val) if n_bets > 0 else 0

ax.set_title(f'Simulated Equity Curve — {n_bets} OOS samples @ {win_rate:.1%} win rate')
ax.set_xlabel('Sample #'); ax.set_ylabel('Balance ($)')
ax.legend()
plt.tight_layout()
plt.savefig('03_equity_curve.png', dpi=120)
plt.show()

print(f"Final balance: ${ce.balance:.2f}")
print(f"Cooling down events: {len(cooling_periods)}")
"""
            code_cell.source = new_code
            break

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Updated 03_walk_forward_results.ipynb equity curve cell")

if __name__ == '__main__':
    update_notebook_03()
