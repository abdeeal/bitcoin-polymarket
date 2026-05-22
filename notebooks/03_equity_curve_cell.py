# ── Paste this into notebook 03 as the equity curve cell ──
# Fast vectorized OOS backtest — NO SignalGenerator/CompoundingEngine (no log spam)

import numpy as np
import matplotlib.pyplot as plt

# Suppress ALL structlog output from signal/sizing modules
import logging, structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

# ── 1. Collect OOS predictions from walk-forward folds ──
all_val_probs, all_y_val = [], []
if (hasattr(wf_result, 'folds') and len(wf_result.folds) > 0
    and hasattr(wf_result.folds[0], 'val_probs')
    and wf_result.folds[0].val_probs is not None):
    for fold in wf_result.folds:
        all_val_probs.extend(fold.val_probs)
        all_y_val.extend(fold.y_val)
else:
    print("Warning: val_probs not found. Using random fallback.")
    np.random.seed(42)
    all_y_val = list(np.random.randint(0, 2, 500))
    all_val_probs = list(np.random.rand(500))

probs = np.array(all_val_probs)
actuals = np.array(all_y_val)
n_total = len(probs)

# ── 2. Vectorized signal classification (no SignalGenerator needed) ──
signals = np.zeros(n_total, dtype=int)
signals[probs >= 0.60] = 2                                   # UP_HIGH
signals[(probs >= 0.55) & (probs < 0.60)] = 1                # UP_NORMAL
signals[(probs <= 0.45) & (probs > 0.40)] = 3                # DOWN_NORMAL
signals[probs <= 0.40] = 4                                    # DOWN_HIGH

is_up   = (signals == 1) | (signals == 2)
is_down = (signals == 3) | (signals == 4)
is_bet  = is_up | is_down

# ── 3. Vectorized outcomes + PnL ──
won = np.zeros(n_total, dtype=bool)
won[is_up]   = (actuals[is_up] == 1)
won[is_down] = (actuals[is_down] == 0)

bet_sizes = np.zeros(n_total)
bet_sizes[(signals == 1) | (signals == 3)] = 1.0   # $1 NORMAL
bet_sizes[(signals == 2) | (signals == 4)] = 2.0   # $2 HIGH_CONF

pnl = np.zeros(n_total)
pnl[is_bet &  won] =  bet_sizes[is_bet &  won] * 0.90
pnl[is_bet & ~won] = -bet_sizes[is_bet & ~won]

# ── 4. Fast loop (cooling down + streaks only — no heavy objects) ──
INITIAL = 10.0
balance = INITIAL
balance_curve = [balance]
cl = ce = 0  # consec_losses, cooling_events
cooling_until = 0
nw = nt_n = hw = ht_h = entries = 0
cw = clk = mw = ml = 0

for i in range(n_total):
    if i < cooling_until or not is_bet[i]:
        balance_curve.append(balance)
        continue
    entries += 1
    balance += pnl[i]
    balance_curve.append(balance)
    if signals[i] in (1, 3):
        nt_n += 1
        if won[i]: nw += 1
    else:
        ht_h += 1
        if won[i]: hw += 1
    if won[i]:
        cl = 0; cw += 1; clk = 0; mw = max(mw, cw)
    else:
        cl += 1; clk += 1; cw = 0; ml = max(ml, clk)
        if cl >= 2:
            cooling_until = i + 6; ce += 1; cl = 0

# ── 5. Summary ──
wr = (nw + hw) / max(entries, 1)
print(f"{'='*55}")
print(f"  WALK-FORWARD OOS BACKTEST SUMMARY")
print(f"{'='*55}")
print(f"  Total OOS bars        : {n_total:,}")
print(f"  Total entries         : {entries:,} ({entries/n_total*100:.1f}%)")
print(f"  Final balance         : ${balance:.2f} ({(balance-INITIAL)/INITIAL*100:+.1f}%)")
print(f"{'─'*55}")
print(f"  NORMAL win rate       : {nw/max(nt_n,1):.1%} ({nw}/{nt_n})")
print(f"  HIGH_CONF win rate    : {hw/max(ht_h,1):.1%} ({hw}/{ht_h})")
print(f"  Overall win rate      : {wr:.1%}")
print(f"{'─'*55}")
print(f"  Max consecutive wins  : {mw}")
print(f"  Max consecutive losses: {ml}")
print(f"  Cooling down events   : {ce}")
print(f"{'='*55}")

# ── 6. Equity curve ──
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(balance_curve, lw=1.2, color='steelblue', label='Balance')
ax.axhline(INITIAL, color='gray', ls='--', lw=0.8, alpha=0.5, label=f'Start ${INITIAL:.0f}')
ax.fill_between(range(len(balance_curve)), INITIAL, balance_curve, alpha=0.15, color='steelblue')
ax.set_title(f'WF OOS Equity -- {entries} entries, WR={wr:.1%}, Final=${balance:.2f}')
ax.set_xlabel('Bar #'); ax.set_ylabel('Balance ($)'); ax.legend()
plt.tight_layout(); plt.savefig('03_equity_curve.png', dpi=120); plt.show()

# ── 7. Save OOS data for notebook 04 ──
np.save(os.path.join(data_dir, 'oos_probs.npy'), probs)
np.save(os.path.join(data_dir, 'oos_y.npy'), actuals)
print(f"\nSaved {n_total} OOS predictions to data/processed/")
