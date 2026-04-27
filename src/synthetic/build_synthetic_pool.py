"""
Build synthetic training pool for RL agent.

Pipeline (per pool generation):
    1. Sample N synthetic paths from the regime-DCC-GARCH-copula generator
    2. For each path:
       a. Compute synthetic-derivable features (tactical, wavelet, cross-asset)
       b. Align real Kronos features by regime matching
       c. Concatenate into a (n_steps, n_features) array
    3. Stack all paths into (n_paths, n_steps, n_features) tensor
    4. Save as compressed npz file

Output schema:
    features:     (n_paths, n_steps, n_features) float32 — full feature tensor
    feature_names: (n_features,) string array — column name lookup
    returns:      (n_paths, n_steps, n_assets) float32 — daily returns per asset
    prices:       (n_paths, n_steps+1, n_assets) float32 — synthetic prices
    volumes:      (n_paths, n_steps, n_assets) float32 — synthetic volumes
    regimes:      (n_paths, n_steps) int8 — regime labels per path

Usage
-----
    from build_synthetic_pool import build_pool
    
    build_pool(
        n_paths=2000,
        n_steps=252,
        seed=42,
        regime_models=...,      # output of fit_per_regime
        trans_mat=...,          # transition matrix
        real_df=...,            # real merged DataFrame for Kronos alignment
        real_regimes=...,       # per-real-date regime labels
        save_to='data/synthetic_pool_seed42.npz',
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time


def build_pool(
    n_paths,
    n_steps,
    seed,
    regime_models,
    trans_mat,
    assets,
    real_df,
    real_regimes,
    kronos_columns,
    feature_builder,           # callable: (returns_path, volumes_path, prices_path) -> DataFrame
    kronos_strategy='regime_match',
    save_to=None,
    verbose=True,
):
    """
    Generate a synthetic training pool with full features.

    Parameters
    ----------
    n_paths, n_steps : int
    seed : int
    regime_models, trans_mat, assets
        Output of your synthetic data fitting pipeline.
    real_df : pd.DataFrame
        Real-data merged DataFrame (for Kronos features).
    real_regimes : np.ndarray, shape (n_real_dates,)
        Regime labels per real date.
    kronos_columns : list[str]
        Column names in real_df that are Kronos features.
    feature_builder : callable
        Function that takes (returns_path, volumes_path, prices_path) for ONE path
        and returns a DataFrame of synthetic-derivable features (no Kronos columns).
        You'll write this — it wraps your tactical + wavelet + cross-asset modules.
    kronos_strategy : 'random' or 'regime_match'
    save_to : str or Path, optional

    Returns
    -------
    pool : dict with keys 'features', 'feature_names', 'returns', etc.
    """
    from kronos_aligner import KronosAligner
    # Assume you have these from your existing code:
    from regime_dcc_garch_copula_V1 import simulate_hybrid_paths

    if verbose:
        print(f"\nBuilding synthetic pool: {n_paths} paths × {n_steps} steps, seed={seed}")
        t0 = time.time()

    # Step 1: simulate paths
    if verbose:
        print(f"  [1/3] Simulating {n_paths} paths...")
    all_returns, all_regimes, all_volumes, all_prices = simulate_hybrid_paths(
        regime_models=regime_models,
        trans_mat=trans_mat,
        assets=assets,
        initial_regime="random",
        n_steps=n_steps,
        n_paths=n_paths,
        return_volumes=True,
        return_prices=True,
        seed=seed,
    )
    if verbose:
        print(f"      returns: {all_returns.shape}, "
              f"prices: {all_prices.shape}, regimes: {all_regimes.shape}")

    # Step 2: Kronos aligner
    if verbose:
        print(f"  [2/3] Setting up Kronos aligner...")
    aligner = KronosAligner(
        real_features_df=real_df,
        real_regime_seq=real_regimes,
        kronos_columns=kronos_columns,
        seed=seed,
    )

    # Step 3: build features per path
    if verbose:
        print(f"  [3/3] Building features for each path...")

    feature_names = None
    features_list = []
    match_pcts = []

    for p in range(n_paths):
        # Synthetic-derivable features (tactical + wavelet + cross-asset)
        synth_features_df = feature_builder(
            all_returns[p], all_volumes[p], all_prices[p]
        )

        # Real Kronos features (regime-matched)
        kronos_block = aligner.assign(
            all_regimes[p],
            strategy=kronos_strategy,
            top_k=5,
            sample_temperature=0.5,
        )
        if kronos_block.attrs.get('match_pct') is not None:
            match_pcts.append(kronos_block.attrs['match_pct'])

        # Combine
        combined = pd.concat([synth_features_df.reset_index(drop=True),
                              kronos_block.reset_index(drop=True)], axis=1)

        # Capture feature names from first path
        if feature_names is None:
            feature_names = list(combined.columns)

        # Sanity: verify column order is consistent across paths
        if list(combined.columns) != feature_names:
            raise ValueError(
                f"Path {p}: feature columns differ from first path. "
                f"First few mismatches: "
                f"{[c for c in combined.columns if c not in feature_names][:3]}"
            )

        features_list.append(combined.values.astype(np.float32))

        if verbose and (p + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (p + 1) / elapsed
            eta = (n_paths - p - 1) / rate
            print(f"      path {p+1}/{n_paths} — {rate:.1f} paths/sec, ETA {eta:.0f}s")

    # Stack
    features_array = np.stack(features_list)  # (n_paths, n_steps, n_features)
    feature_names_array = np.array(feature_names)

    if verbose:
        elapsed = time.time() - t0
        print(f"\n  Feature tensor: {features_array.shape}, "
              f"dtype={features_array.dtype}, "
              f"size={features_array.nbytes / 1e6:.1f} MB")
        if match_pcts:
            print(f"  Kronos match quality: mean={np.mean(match_pcts):.1%}, "
                  f"min={np.min(match_pcts):.1%}, max={np.max(match_pcts):.1%}")
        print(f"  Total build time: {elapsed:.1f}s")

    pool = {
        'features': features_array,
        'feature_names': feature_names_array,
        'returns': all_returns.astype(np.float32),
        'prices': all_prices.astype(np.float32),
        'volumes': all_volumes.astype(np.float32),
        'regimes': all_regimes.astype(np.int8),
        'metadata': np.array([{
            'seed': seed,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'kronos_strategy': kronos_strategy,
            'kronos_match_mean_pct': float(np.mean(match_pcts)) if match_pcts else None,
        }], dtype=object),
    }

    # Save
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_to, **pool)
        if verbose:
            file_mb = save_to.stat().st_size / 1e6
            print(f"  Saved to: {save_to} ({file_mb:.1f} MB)")

    return pool


# -------------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------------
def load_pool(path):
    """
    Load a synthetic pool. Returns a dict with the same structure as build_pool.
    """
    data = np.load(path, allow_pickle=True)
    return {
        'features': data['features'],
        'feature_names': data['feature_names'],
        'returns': data['returns'],
        'prices': data['prices'],
        'volumes': data['volumes'],
        'regimes': data['regimes'],
        'metadata': data['metadata'][0] if 'metadata' in data else {},
    }


# -------------------------------------------------------------------------
# Sampling helper for RL training loop
# -------------------------------------------------------------------------
class PoolSampler:
    """
    Wrap a synthetic pool for batched access during training.

    Usage:
        pool = load_pool('data/synthetic_pool_seed42.npz')
        sampler = PoolSampler(pool)
        
        for batch_idx in range(n_batches):
            batch = sampler.sample(batch_size=128, rng=rng)
            features = batch['features']    # (128, n_steps, n_features)
            returns  = batch['returns']     # (128, n_steps, n_assets)
            # ... train RL agent
    """

    def __init__(self, pool):
        self.pool = pool
        self.n_paths = pool['features'].shape[0]

    def sample(self, batch_size, rng=None, replace=False):
        """Sample a batch of paths. Returns dict of arrays sliced by path."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(self.n_paths, size=batch_size, replace=replace)
        return {
            'features': self.pool['features'][idx],
            'returns':  self.pool['returns'][idx],
            'prices':   self.pool['prices'][idx],
            'volumes':  self.pool['volumes'][idx],
            'regimes':  self.pool['regimes'][idx],
            'path_idx': idx,
        }
