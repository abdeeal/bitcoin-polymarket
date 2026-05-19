"""
Bootstrap Training Script — Download 90 days of historical data and train all models.

Usage:
    python -m src.training.bootstrap

This script:
  1. Downloads 90 days of 1-minute BTC candles from Binance (public API, no key needed)
  2. Downloads ETH and SOL candles for cross-asset features
  3. Fills a RingBuffer and computes all 150 features for each 5-minute bar
  4. Runs the full TrainingPipeline (walk-forward validation + final training)
  5. Saves model weights to models/latest/

After running this, restart the bot and it will auto-load the trained models.
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


async def download_historical_candles(days: int = 120) -> list[dict]:
    """
    Download `days` worth of 1-minute BTC candles from Binance.
    Also fetches ETH and SOL for cross-asset features.
    
    Returns list of bar dicts ordered oldest → newest.
    """
    from src.ingestion.binance_rest import BinanceRESTClient

    client = BinanceRESTClient()
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    
    total_candles = days * 24 * 60  # 1-minute candles
    
    print(f"\n{'='*60}")
    print(f"  BOOTSTRAP TRAINING — Downloading {days} days of data")
    print(f"  Expected candles: ~{total_candles:,}")
    print(f"  Date range: {datetime.fromtimestamp(start_time/1000, tz=timezone.utc).strftime('%Y-%m-%d')} → {datetime.fromtimestamp(end_time/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")
    
    # Download BTC candles
    print("[1/3] Downloading BTC/USDT 1m candles...")
    btc_candles = await client.fetch_klines(
        symbol="BTCUSDT",
        interval="1m",
        limit=total_candles,
        start_time=start_time,
        end_time=end_time,
    )
    print(f"  ✓ Downloaded {len(btc_candles):,} BTC candles")
    
    # Download ETH candles for cross-asset features
    print("[2/3] Downloading ETH/USDT 1m candles...")
    eth_candles = await client.fetch_klines(
        symbol="ETHUSDT",
        interval="1m",
        limit=total_candles,
        start_time=start_time,
        end_time=end_time,
    )
    print(f"  ✓ Downloaded {len(eth_candles):,} ETH candles")
    
    # Download SOL candles for cross-asset features
    print("[3/3] Downloading SOL/USDT 1m candles...")
    sol_candles = await client.fetch_klines(
        symbol="SOLUSDT",
        interval="1m",
        limit=total_candles,
        start_time=start_time,
        end_time=end_time,
    )
    print(f"  ✓ Downloaded {len(sol_candles):,} SOL candles")
    
    await client.close()
    
    # Merge ETH and SOL data into BTC candles by timestamp
    eth_map = {c['timestamp']: c for c in eth_candles}
    sol_map = {c['timestamp']: c for c in sol_candles}
    
    for bar in btc_candles:
        ts = bar['timestamp']
        eth = eth_map.get(ts, {})
        sol = sol_map.get(ts, {})
        bar['eth_close'] = eth.get('eth_close', 0.0)
        bar['eth_volume'] = eth.get('eth_volume', 0.0)
        bar['sol_close'] = sol.get('sol_close', 0.0)
        bar['sol_volume'] = sol.get('sol_volume', 0.0)
    
    return btc_candles


def compute_features_from_candles(candles: list[dict], window_size: int = 200) -> tuple[list[dict], np.ndarray]:
    """
    Compute all 150 features for each bar using a sliding window approach.
    
    Returns:
        (feature_dicts, close_prices) - list of feature snapshots and corresponding close prices
    """
    from src.memory.ring_buffer import RingBuffer
    from src.features.feature_engine import FeatureEngine
    
    print(f"\n[4/6] Computing features for {len(candles):,} candles...")
    print(f"  Window size: {window_size}, will produce ~{len(candles) - window_size:,} feature vectors")
    
    # Create a large ring buffer
    ring_buffer = RingBuffer(capacity=max(len(candles) + 100, 1000))
    feature_engine = FeatureEngine(ring_buffer, window_size=window_size)
    
    feature_list = []
    close_prices = []
    
    t0 = time.perf_counter()
    last_print = 0
    
    for i, bar in enumerate(candles):
        ring_buffer.push(bar)
        
        # Only compute features after we have enough data in the buffer
        # and sample every 5 bars (5-minute equivalent for 1m candles)
        if ring_buffer.size >= window_size and (i % 5 == 0):
            features = feature_engine.compute_sync(bot_state="ACTIVE")
            
            if features and not features.get('_insufficient_data'):
                feature_list.append(features)
                close_prices.append(bar['close'])
        
        # Progress reporting
        progress = int((i / len(candles)) * 100)
        if progress >= last_print + 10:
            elapsed = time.perf_counter() - t0
            print(f"  ... {progress}% ({i:,}/{len(candles):,}) — {elapsed:.1f}s elapsed, {len(feature_list):,} features computed")
            last_print = progress
    
    elapsed = time.perf_counter() - t0
    print(f"  ✓ Computed {len(feature_list):,} feature vectors in {elapsed:.1f}s")
    
    return feature_list, np.array(close_prices)


def run_training(feature_list: list[dict], close_prices: np.ndarray, save_dir: str = "models/latest") -> dict:
    """
    Run the full training pipeline.
    """
    from src.training.train_pipeline import TrainingPipeline
    
    print(f"\n[5/6] Preparing feature matrix...")
    
    pipeline = TrainingPipeline(model_save_dir=save_dir)
    
    # Convert feature dicts to numpy matrix
    X, feature_names = pipeline.prepare_features(feature_list)
    
    print(f"  ✓ Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  ✓ Close prices: {len(close_prices):,} values")
    print(f"  ✓ Feature names: {feature_names[:5]}... ({len(feature_names)} total)")
    
    # Run training
    print(f"\n[6/6] Training models (walk-forward + final)...")
    print(f"  This may take several minutes...\n")
    
    results = pipeline.train_from_features(
        X=X,
        close_prices=close_prices,
        feature_names=feature_names,
        run_walk_forward=True,
        save=True,
    )
    
    return results


async def main():
    """Main bootstrap training pipeline."""
    overall_t0 = time.perf_counter()
    
    # Step 1-3: Download data
    candles = await download_historical_candles(days=90)
    
    if len(candles) < 5000:
        print(f"\n❌ ERROR: Only got {len(candles)} candles. Need at least 5000 for training.")
        sys.exit(1)
    
    # Step 4: Compute features
    feature_list, close_prices = compute_features_from_candles(candles)
    
    if len(feature_list) < 1000:
        print(f"\n❌ ERROR: Only computed {len(feature_list)} feature vectors. Need at least 1000.")
        sys.exit(1)
    
    # Step 5-6: Train models
    save_dir = "models/latest"
    results = run_training(feature_list, close_prices, save_dir=save_dir)
    
    # Summary
    elapsed = time.perf_counter() - overall_t0
    
    print(f"\n{'='*60}")
    print(f"  ✅ BOOTSTRAP TRAINING COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  Models saved to: {save_dir}/")
    print(f"{'='*60}")
    
    if 'walk_forward' in results:
        wf = results['walk_forward']
        print(f"\n  Walk-Forward Results:")
        print(f"    Mean Accuracy: {wf.get('mean_accuracy', 'N/A')}")
        print(f"    Folds:         {wf.get('n_folds', 'N/A')}")
    
    if 'final_training' in results:
        ft = results['final_training']
        print(f"\n  Final Training:")
        print(f"    Train time: {ft.get('total_train_time_sec', 'N/A')}s")
        print(f"    LGBM fitted: {ft.get('lgbm', {}).get('n_features', 'N/A')} features")
    
    print(f"\n  Next step: Restart the bot with 'python -m src.main'")
    print(f"  The bot will automatically load the trained models.\n")


if __name__ == "__main__":
    asyncio.run(main())
