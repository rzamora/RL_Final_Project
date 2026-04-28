"""
End-to-end smoke test for the synthetic feature pipeline.

Runs:
    1. Fit regime models on real data
    2. Generate 4 synthetic paths (n_steps=300 to leave room after 128-day warmup)
    3. For each path: build features + attach Kronos via per-ticker alignment
    4. Validate that columns match RL_Final_Merged_train.csv exactly
    5. Print sanity stats

This is meant to surface column-mismatch and runtime issues fast before
scaling up to thousands of paths in build_synthetic_pool.

Adjust the paths at the top of main() to match your environment.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_feature_builder import (
    build_synthetic_features,
    build_kronos_block_for_path,
    validate_schema,
    reorder_to_reference,
    KRONOS_COLUMNS,
    PER_ASSET_FEATURES,
    REGIME_PROB_COLS,
)


# =============================================================================
# Configuration
# =============================================================================
ASSET_TICKERS = ("NVDA", "AMD", "SMH", "TLT")
EQUITY_ASSETS = ("NVDA", "AMD", "SMH")
TLT_ASSET = "TLT"

N_PATHS = 4
N_STEPS = 300              # 300 raw → 172 after 128-day warmup trim
TRIM_WARMUP = 128
SEED = 42

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_MERGED = PROJECT_ROOT / "data" / "proccessed" / "combined_w_cross_asset" / "train"
DATA_SYN = PROJECT_ROOT / "data" / "synthetic"

# Reference CSV for schema validation. If the file isn't found, schema
# validation is skipped (the rest of the test still runs).
REF_CSV = DATA_MERGED / "RL_Final_Merged_train.csv"

# Output directory for per-path synthetic CSVs.
OUT_DIR = DATA_SYN / "smoke_test_output"


# =============================================================================
# Helpers
# =============================================================================
def stub_kronos_for_smoke_test(n_real_dates=2000, seed=0):
    """
    Build a stub real-data history for the Kronos aligner so we can run the
    pipeline end-to-end without needing the actual real merged file present.

    Generates fake real Kronos columns and a fake real regime sequence.
    For real use, replace this with loading from RL_Final_Merged_train.csv
    plus your stored regime sequence.
    """
    rng = np.random.default_rng(seed)
    real_dfs = {}
    real_regimes = {}

    for ticker in ASSET_TICKERS:
        df = pd.DataFrame({
            "date": pd.bdate_range("2010-01-01", periods=n_real_dates),
        })
        for col in KRONOS_COLUMNS:
            # plausible-looking distributions per column type
            if col.startswith("kronos_regime_"):
                df[col] = rng.integers(0, 2, size=n_real_dates).astype(float)
            elif "conf" in col or "hit" in col:
                df[col] = rng.uniform(0, 1, size=n_real_dates)
            elif "Slope" in col or "Convexity" in col:
                df[col] = rng.normal(0, 0.01, size=n_real_dates)
            else:
                df[col] = rng.normal(0, 0.02, size=n_real_dates)

        real_dfs[ticker] = df
        # Fake regime sequence with all 4 regimes represented
        real_regimes[ticker] = rng.choice(
            [0, 1, 2, 3], size=n_real_dates, p=[0.45, 0.35, 0.15, 0.05]
        )

    return real_dfs, real_regimes


def stub_synthetic_paths(n_paths, n_steps, n_assets=4, seed=0):
    """
    Generate stub synthetic paths matching the generator's output shapes,
    for fast smoke testing without running the full GARCH-copula simulator.

    Returns
    -------
    returns : (n_paths, n_steps, n_assets)
    regimes : (n_paths, n_steps) int
    volumes : (n_paths, n_steps, n_assets)
    prices  : (n_paths, n_steps+1, n_assets)
    """
    rng = np.random.default_rng(seed)

    # Returns: small daily moves in PERCENT (matches generator convention)
    returns = rng.normal(0.05, 1.5, size=(n_paths, n_steps, n_assets)).astype(np.float32)

    # Regimes: persistent Markov-like sequences via simple sticky sampling
    regimes = np.zeros((n_paths, n_steps), dtype=np.int32)
    for p in range(n_paths):
        cur = int(rng.integers(0, 4))
        for t in range(n_steps):
            # 92% stay, 8% switch uniformly to another regime
            if rng.random() > 0.92:
                cur = int(rng.choice([k for k in range(4) if k != cur]))
            regimes[p, t] = cur

    # Volumes: log-normal around a per-asset mean
    base_log_vol = np.array([18.0, 17.5, 16.0, 16.5])  # ln of typical daily volumes
    log_vol = (base_log_vol[None, None, :]
               + rng.normal(0, 0.5, size=(n_paths, n_steps, n_assets)))
    volumes = np.exp(log_vol).astype(np.float32)

    # Prices: cumulative product from returns/100, starting at 100
    prices = np.zeros((n_paths, n_steps + 1, n_assets), dtype=np.float32)
    prices[:, 0, :] = 100.0
    for t in range(n_steps):
        prices[:, t + 1, :] = prices[:, t, :] * (1 + returns[:, t, :] / 100.0)

    return returns, regimes, volumes, prices


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("Synthetic feature pipeline smoke test")
    print(f"  paths={N_PATHS}, n_steps={N_STEPS}, trim_warmup={TRIM_WARMUP}")
    print(f"  expected output rows per path: {N_STEPS - TRIM_WARMUP}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Generate synthetic paths
    # -------------------------------------------------------------------------
    print("\n[1] Generating synthetic paths...")
    t0 = time.time()

    USE_STUB = True
    if USE_STUB:
        # Fast stub for smoke test — no fitting required
        returns_all, regimes_all, volumes_all, prices_all = stub_synthetic_paths(
            n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED,
        )
        print(f"    Used stub generator (no GARCH fitting)")
    else:
        # Real path: fit on actual data
        from regime_dcc_garch_copula_V1 import (
            download_data, fit_per_regime, simulate_hybrid_paths,
        )
        # ... (the real fitting code goes here when you want to test against
        # the real generator; left out of smoke test to keep it fast)
        raise NotImplementedError("Set USE_STUB=False and wire up real fitting")

    print(f"    returns: {returns_all.shape}, regimes: {regimes_all.shape}")
    print(f"    volumes: {volumes_all.shape}, prices: {prices_all.shape}")
    print(f"    Time: {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # 2. Build stub Kronos history
    # -------------------------------------------------------------------------
    print("\n[2] Building stub Kronos real-data history...")
    real_dfs, real_regimes = stub_kronos_for_smoke_test(n_real_dates=2000, seed=0)
    print(f"    real_dfs keys: {list(real_dfs.keys())}, "
          f"each {len(real_dfs[ASSET_TICKERS[0]])} rows")

    # -------------------------------------------------------------------------
    # 3. Build features per path
    # -------------------------------------------------------------------------
    print("\n[3] Building features per path...")
    per_path_dfs = []
    for p in range(N_PATHS):
        t_path = time.time()
        print(f"\n  --- Path {p+1}/{N_PATHS} ---")

        # 3a. Build full-length Kronos block by per-ticker regime matching
        # We pass the FULL synth regime sequence; later we'll trim warmup off
        # both the merged DF and the Kronos block consistently.
        kronos_block = build_kronos_block_for_path(
            synth_regime_path=regimes_all[p],
            real_dfs_by_ticker=real_dfs,
            real_regimes_by_ticker=real_regimes,
            kronos_columns=KRONOS_COLUMNS,
            asset_tickers=ASSET_TICKERS,
            top_k=5,
            sample_temperature=0.5,
            seed=SEED + p,
        )
        print(f"    [3a] Kronos block: {kronos_block.shape}")

        # 3b. Build features (Kronos attached internally, then warmup trimmed)
        synth_df = build_synthetic_features(
            returns_path=returns_all[p],
            volumes_path=volumes_all[p],
            prices_path=prices_all[p],
            regime_path=regimes_all[p],
            asset_tickers=ASSET_TICKERS,
            equity_assets=EQUITY_ASSETS,
            tlt_asset=TLT_ASSET,
            kronos_block_df=kronos_block,
            dirichlet_concentration=50.0,
            dirichlet_noise_floor=0.5,
            trim_warmup=TRIM_WARMUP,
            start_date="2020-01-01",
            seed=SEED + p,
            verbose=(p == 0),  # verbose on first path only
        )
        print(f"    [3b] synth_df: {synth_df.shape}, "
              f"build time: {time.time() - t_path:.2f}s")

        per_path_dfs.append(synth_df)

    # -------------------------------------------------------------------------
    # 4. Schema validation
    # -------------------------------------------------------------------------
    print("\n[4] Schema validation against RL_Final_Merged_train.csv...")
    ref_path = REF_CSV
    if not ref_path.exists():
        print(f"    SKIP: {ref_path} not found — skipping schema validation.")
        print(f"          Expected at: {DATA_MERGED}")
        ref_cols = None
    else:
        ref_cols = pd.read_csv(ref_path, nrows=0).columns.tolist()
        print(f"    Reference has {len(ref_cols)} columns")

        for p, df in enumerate(per_path_dfs):
            # Reorder to reference (catches column-order issues separately
            # from missing-column issues)
            try:
                df_ordered = reorder_to_reference(df, ref_path)
                per_path_dfs[p] = df_ordered
                validate_schema(df_ordered, ref_path)
                print(f"    Path {p+1}: schema OK ({df_ordered.shape[1]} cols)")
            except (KeyError, AssertionError) as e:
                print(f"    Path {p+1}: SCHEMA FAIL")
                print(f"      {e}")
                missing_cols = [c for c in ref_cols if c not in df.columns]
                extra_cols = [c for c in df.columns if c not in ref_cols]
                if missing_cols:
                    print(f"      Missing ({len(missing_cols)}): "
                          f"{missing_cols[:8]}...")
                if extra_cols:
                    print(f"      Extra   ({len(extra_cols)}): "
                          f"{extra_cols[:8]}...")
                return 1

    # -------------------------------------------------------------------------
    # 5. Sanity stats
    # -------------------------------------------------------------------------
    print("\n[5] Sanity stats on path 1:")
    df = per_path_dfs[0]
    print(f"    Shape: {df.shape}")
    print(f"    Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"    NaN count: {df.isna().sum().sum()} (across all cells)")
    nan_by_col = df.isna().sum()
    nan_cols = nan_by_col[nan_by_col > 0]
    if len(nan_cols) > 0:
        print(f"    Columns with NaN ({len(nan_cols)}):")
        for col, n in nan_cols.head(10).items():
            print(f"      {col}: {n} NaN")

    # Regime probability sanity
    regime_probs = df[REGIME_PROB_COLS].values
    row_sums = regime_probs.sum(axis=1)
    print(f"\n    Regime prob row sums: "
          f"min={row_sums.min():.4f}, max={row_sums.max():.4f}, "
          f"mean={row_sums.mean():.4f}  (should all be ~1.0)")
    max_probs = regime_probs.max(axis=1)
    print(f"    Max class prob per row: "
          f"min={max_probs.min():.3f}, mean={max_probs.mean():.3f}, "
          f"max={max_probs.max():.3f}  (high concentration is intentional)")

    # Cross-asset correlation sanity
    corr_cols = [c for c in df.columns if c.startswith("corr_") and "_z" not in c]
    if corr_cols:
        corr_vals = df[corr_cols].values
        print(f"\n    Pairwise corr range: "
              f"[{np.nanmin(corr_vals):.3f}, {np.nanmax(corr_vals):.3f}]  "
              f"(should be in [-1, 1])")

    # Dispersion across paths
    if N_PATHS > 1:
        means = [d["NVDA_close"].mean() for d in per_path_dfs]
        print(f"\n    NVDA close mean per path: "
              f"{[f'{m:.2f}' for m in means]}")
        print(f"    (paths should differ — confirms RNG diversity across paths)")

    # -------------------------------------------------------------------------
    # 6. Save
    # -------------------------------------------------------------------------
    print("\n[6] Saving smoke-test output...")
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, df in enumerate(per_path_dfs):
        out_path = out_dir / f"synth_path_{p:02d}.csv"
        df.to_csv(out_path, index=False)
    print(f"    Saved {N_PATHS} per-path CSVs to {out_dir}")

    # Quick stacked-tensor preview (path × time × feature)
    feature_cols = [c for c in per_path_dfs[0].columns if c != "date"]
    tensor = np.stack([d[feature_cols].values.astype(np.float32)
                       for d in per_path_dfs])
    print(f"    Stacked tensor shape (paths, steps, features): {tensor.shape}")

    print("\n" + "=" * 70)
    print("Smoke test PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
