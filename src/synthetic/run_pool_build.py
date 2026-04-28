"""
Production runner for the synthetic feature pool.

Replaces the stub-based smoke test with real data:
    - Real path simulation via simulate_hybrid_paths (loaded from fitted pickle)
    - Real Kronos features sliced from RL_Final_Merged_train.csv per ticker

Two modes via command line:
    python run_pool_build.py test          # 50 paths, schema validation, round-trip
    python run_pool_build.py production    # 2000 paths, full save

Defaults are tuned for: n_steps=512, trim_warmup=128 → 384 clean rows per path
(roughly 18 months of trading days). If you want shorter clean episodes,
increase trim_warmup; if you want longer, increase n_steps.

Edit the PATHS section to match your environment.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_synthetic_pool_v2 import (
    build_pool,
    load_pool,
    verify_round_trip,
    PoolSampler,
)
from synthetic_feature_builder import REGIME_PROB_COLS


# =============================================================================
# Paths — set DATA_ROOT to override, or pass --data-root on the CLI
# =============================================================================
# Resolution order (first hit wins):
#   1. --data-root CLI arg
#   2. DATA_ROOT environment variable
#   3. DEFAULT_DATA_ROOT below — EDIT THIS to match your machine
#
# DATA_ROOT is the parent of the `data/` directory, e.g. if your data lives
# at /Users/you/proj/data/synthetic/..., set DATA_ROOT = /Users/you/proj.
import os

DEFAULT_DATA_ROOT = Path("/Users/rafael/Documents/GitHub/RL_Final_Project")


def resolve_paths(data_root):
    """Build the input/output path set from a single data root."""
    data_root = Path(data_root).expanduser().resolve()
    return {
        "data_root": data_root,
        "fitted_pickle": data_root / "data" / "synthetic" / "models"
                         / "synthetic_generator_FITTED.pkl",
        "merged_train_csv": data_root / "data" / "proccessed"
                            / "combined_w_cross_asset" / "train"
                            / "RL_Final_Merged_train.csv",
        "pool_out_dir": data_root / "data" / "synthetic" / "pools",
        "checkpoint_dir": data_root / "data" / "synthetic" / "pool_checkpoints",
    }


def _suggest_pickle_locations(data_root):
    """
    On a missing-pickle error, scan a few likely directories for any *.pkl
    files and print them as suggestions. Saves the user from having to run
    `find` themselves.
    """
    candidates = [
        data_root / "data" / "synthetic",
        data_root / "data" / "synthetic" / "models",
        data_root / "data",
    ]
    found = []
    for d in candidates:
        if not d.exists():
            continue
        try:
            found.extend(sorted(d.rglob("*.pkl")))
        except (PermissionError, OSError):
            continue

    if not found:
        print(f"\n  (No .pkl files found under {data_root}/data — "
              f"the fitted pickle has likely never been generated.)")
        return

    print(f"\n  Found {len(found)} .pkl file(s) under {data_root}/data — "
          f"any of these might be the one you want:")
    for p in found[:10]:
        try:
            size_mb = p.stat().st_size / 1e6
            print(f"    --fitted-pickle {p}   ({size_mb:.1f} MB)")
        except OSError:
            print(f"    --fitted-pickle {p}")


# =============================================================================
# Pool config presets
# =============================================================================
ASSET_TICKERS = ("NVDA", "AMD", "SMH", "TLT")
EQUITY_ASSETS = ("NVDA", "AMD", "SMH")
TLT_ASSET = "TLT"

# n_steps / trim_warmup decision:
#   - Wavelet warmup forces trim >= 128 to avoid NaN rows.
#   - clean_rows = n_steps - trim_warmup.
#   - For RL training we want long enough episodes that the agent sees full
#     regime cycles. A typical regime persists ~20-60 days, so 250-500 clean
#     rows lets the agent see ~5-15 regime transitions per episode.
#   - 512 / 128 → 384 clean rows ≈ 1.5 trading years. Good default.
#   - If you want closer to 1 trading year, use 380 / 128 → 252 clean rows.
#   - Larger n_steps costs more sim time but improves Kronos match quality
#     (more regime sequence to match against).

PRESETS = {
    "test": dict(
        n_paths=50,
        n_steps=512,
        trim_warmup=128,
        seed=42,
    ),
    "production": dict(
        # n_steps=584, trim_warmup=200 → 384 clean rows per path.
        # The 200-row trim clears the deepest chained-rolling warmup
        # (corr_w60_z120 needed ~178 rows in the seed=42 pool to stop
        # producing NaN). 22 rows of margin handles seed-to-seed variance.
        # Bumping n_steps from 512 to 584 keeps the same 384 clean rows we
        # had at trim_warmup=128, so episode length doesn't shrink.
        n_paths=2000,
        n_steps=584,
        trim_warmup=200,
        seed=43,
    ),
    "small_demo": dict(
        # Quick build to verify everything wires up — finishes in seconds
        n_paths=4,
        n_steps=300,
        trim_warmup=128,
        seed=42,
    ),
}


# =============================================================================
# Sanity diagnostics on a built pool
# =============================================================================
def run_sanity_checks(pool, asset_tickers=ASSET_TICKERS):
    """Print diagnostics on a built pool — call after build_pool returns."""
    print("\n" + "=" * 70)
    print("Pool sanity checks")
    print("=" * 70)

    features = pool["features"]
    feature_names = list(pool["feature_names"])
    n_paths, n_steps, n_features = features.shape

    print(f"\nShape: {n_paths} paths × {n_steps} steps × {n_features} features")

    # NaN check
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    print(f"\nNaN in features:  {n_nan} / {features.size} cells "
          f"({100 * n_nan / features.size:.4f}%)")
    print(f"Inf in features:  {n_inf}")
    if n_nan > 0:
        # Locate which columns have NaNs
        nan_per_col = np.isnan(features).sum(axis=(0, 1))
        bad_cols = [(feature_names[i], int(nan_per_col[i]))
                    for i in range(n_features) if nan_per_col[i] > 0]
        print(f"Columns with NaN ({len(bad_cols)}):")
        for col, n in bad_cols[:15]:
            print(f"  {col:<40s} {n} NaN")

    # Regime probability sanity
    if all(c in feature_names for c in REGIME_PROB_COLS):
        idx = [feature_names.index(c) for c in REGIME_PROB_COLS]
        regime_probs = features[:, :, idx]
        row_sums = regime_probs.sum(axis=2)
        print(f"\nRegime prob row sums: "
              f"min={row_sums.min():.4f}, max={row_sums.max():.4f}, "
              f"mean={row_sums.mean():.4f}  (expect ~1.0)")
        max_per_row = regime_probs.max(axis=2)
        print(f"Max class prob per row: "
              f"mean={max_per_row.mean():.3f}, "
              f"min={max_per_row.min():.3f}  "
              f"(high concentration is intentional from Dirichlet smoothing)")

    # Cross-asset correlation sanity
    corr_cols_idx = [
        i for i, c in enumerate(feature_names)
        if c.startswith("corr_") and "_z" not in c
    ]
    if corr_cols_idx:
        corr_vals = features[:, :, corr_cols_idx]
        finite = np.isfinite(corr_vals)
        if finite.any():
            print(f"\nCross-asset corr columns ({len(corr_cols_idx)}): "
                  f"range [{np.nanmin(corr_vals):.3f}, {np.nanmax(corr_vals):.3f}]  "
                  f"(expect within [-1, 1])")

    # Path diversity (RNG sanity)
    if n_paths > 1:
        # Pick a feature that should differ across paths — first feature is
        # typically a price-like column from one ticker
        per_path_means = features[:, :, 0].mean(axis=1)
        print(f"\nPath diversity check on '{feature_names[0]}':")
        print(f"  mean per path: min={per_path_means.min():.3f}, "
              f"max={per_path_means.max():.3f}, "
              f"std across paths={per_path_means.std():.3f}  "
              f"(should NOT be ~0 — paths must differ)")

    # Returns/regimes alignment check
    returns = pool["returns"]
    regimes = pool["regimes"]
    print(f"\nCompanion arrays:")
    print(f"  returns shape: {returns.shape}, "
          f"per-step mean across paths: {returns.mean():.4f}")
    print(f"  regimes shape: {regimes.shape}, "
          f"unique regimes seen: {sorted(np.unique(regimes).tolist())}")

    # Per-regime fraction in the pool
    n_total = regimes.size
    print(f"  Regime distribution in pool:")
    for r in sorted(np.unique(regimes).tolist()):
        frac = (regimes == r).sum() / n_total
        print(f"    regime {r}: {frac:.1%}")


# =============================================================================
# Round-trip + sampler smoke test
# =============================================================================
def run_round_trip_and_sampler_check(pool, save_path):
    """Save → load → spot-check → sample a batch."""
    print("\n" + "=" * 70)
    print(f"Round-trip + sampler check")
    print("=" * 70)

    reloaded = verify_round_trip(pool, save_path, verbose=True)

    # Sampler smoke
    sampler = PoolSampler(reloaded)
    rng = np.random.default_rng(0)
    sample_size = min(8, reloaded["features"].shape[0])
    batch = sampler.sample(batch_size=sample_size, rng=rng, replace=False)
    print(f"\nSampler.sample(batch_size={sample_size}) shapes:")
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:<14s}: {v.shape} {v.dtype}")


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build synthetic feature pool for RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "preset",
        nargs="?",
        choices=sorted(PRESETS.keys()),
        default="test",
        help="Which preset to run. test=50 paths, production=2000 paths, "
             "small_demo=4 paths (default: test)",
    )
    parser.add_argument(
        "--n-paths", type=int, default=None,
        help="Override preset n_paths",
    )
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="Override preset n_steps (raw, before warmup trim)",
    )
    parser.add_argument(
        "--trim-warmup", type=int, default=None,
        help="Override preset trim_warmup (rows to drop from start)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override preset seed",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="",
        help="Optional suffix appended to output filename",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving (just build + sanity-check in memory)",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint saves during build",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help=f"Override data root (parent of `data/`). Falls back to "
             f"$DATA_ROOT env var, then DEFAULT_DATA_ROOT={DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--fitted-pickle", type=str, default=None,
        help="Direct override of the fitted pickle path (bypasses --data-root)",
    )
    parser.add_argument(
        "--merged-csv", type=str, default=None,
        help="Direct override of the merged train CSV path (bypasses --data-root)",
    )
    args = parser.parse_args()

    # Resolve data root from CLI / env / default
    data_root = (
        args.data_root
        or os.environ.get("DATA_ROOT")
        or DEFAULT_DATA_ROOT
    )
    paths = resolve_paths(data_root)

    # Per-file overrides take precedence over data-root resolution
    fitted_pickle = (
        Path(args.fitted_pickle).expanduser().resolve()
        if args.fitted_pickle else paths["fitted_pickle"]
    )
    merged_train_csv = (
        Path(args.merged_csv).expanduser().resolve()
        if args.merged_csv else paths["merged_train_csv"]
    )
    pool_out_dir = paths["pool_out_dir"]
    checkpoint_dir_default = paths["checkpoint_dir"]

    cfg = dict(PRESETS[args.preset])  # copy
    if args.n_paths is not None:    cfg["n_paths"] = args.n_paths
    if args.n_steps is not None:    cfg["n_steps"] = args.n_steps
    if args.trim_warmup is not None: cfg["trim_warmup"] = args.trim_warmup
    if args.seed is not None:        cfg["seed"] = args.seed

    # Sanity on overrides
    if cfg["trim_warmup"] < 128:
        print(f"WARNING: trim_warmup={cfg['trim_warmup']} is below 128. "
              f"Wavelet features need 128 rows of history; you may see NaNs "
              f"in the first {128 - cfg['trim_warmup']} clean rows.")
    if cfg["n_steps"] - cfg["trim_warmup"] < 60:
        print(f"WARNING: only {cfg['n_steps'] - cfg['trim_warmup']} clean rows "
              f"per path. Episodes may be too short for the RL agent to learn.")

    print(f"\nPreset: {args.preset}")
    print(f"Config: {cfg}")
    print(f"Data root: {paths['data_root']}")
    print(f"Inputs:")
    print(f"  fitted pickle:    {fitted_pickle}")
    print(f"  merged train CSV: {merged_train_csv}")

    # Resolve output path
    if not args.no_save:
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        save_to = (pool_out_dir / f"synthetic_pool_{args.preset}_"
                                  f"n{cfg['n_paths']}_seed{cfg['seed']}{suffix}.npz")
        print(f"  output:           {save_to}")
    else:
        save_to = None
        print(f"  output:           (none — --no-save)")

    # Checkpoint config: only on production by default
    checkpoint_every = None
    checkpoint_dir = None
    if not args.no_checkpoint and args.preset == "production":
        checkpoint_every = 200
        checkpoint_dir = checkpoint_dir_default
        print(f"  checkpoint:       every {checkpoint_every} paths → {checkpoint_dir}")

    # Validate inputs exist before doing the long build
    if not fitted_pickle.exists():
        print(f"\nERROR: Fitted pickle not found at {fitted_pickle}")
        print(f"  Either:")
        print(f"    a) Run regime_dcc_garch_copula_V1.main() to generate it,")
        print(f"    b) Pass --fitted-pickle /path/to/your.pkl, or")
        print(f"    c) Set --data-root or $DATA_ROOT so the default path resolves.")
        _suggest_pickle_locations(paths["data_root"])
        return 1
    if not merged_train_csv.exists():
        print(f"\nERROR: Merged train CSV not found at {merged_train_csv}")
        print(f"  Pass --merged-csv /path/to/RL_Final_Merged_train.csv, or")
        print(f"  set --data-root / $DATA_ROOT so the default path resolves.")
        return 1

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------
    t0 = time.time()
    pool = build_pool(
        fitted_pickle_path=fitted_pickle,
        merged_train_csv_path=merged_train_csv,
        n_paths=cfg["n_paths"],
        n_steps=cfg["n_steps"],
        trim_warmup=cfg["trim_warmup"],
        seed=cfg["seed"],
        asset_tickers=ASSET_TICKERS,
        equity_assets=EQUITY_ASSETS,
        tlt_asset=TLT_ASSET,
        save_to=save_to,
        validate_against_reference=True,
        reorder_columns_to_reference=True,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
    )
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # Sanity
    # -------------------------------------------------------------------------
    run_sanity_checks(pool, asset_tickers=ASSET_TICKERS)

    # Round-trip only on test/small (production save is already verified by
    # the in-memory pool dict matching what was just written)
    if args.preset in ("test", "small_demo") and save_to is not None:
        run_round_trip_and_sampler_check(pool, save_to)

    print("\n" + "=" * 70)
    print(f"DONE — preset={args.preset}, n_paths={cfg['n_paths']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
