"""
Step 1 — Data sanity check.

Verifies:
  - Real CSVs load correctly
  - process_raw_df produces 313 features (closes dropped)
  - No NaN/Inf in features
  - Returns are sensibly sized per asset
  - Synthetic pool loads with matching feature dim (313)
  - DTB3 file is parseable (we'll wire it into the env later)

Run from project root: `python tests/test_01_data.py`
"""

import sys
from pathlib import Path

# Bootstrap: make project root importable regardless of how this is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from project_config import PATHS


def main():
    print("=" * 70)
    print("Step 1 — Data sanity check")
    print("=" * 70)

    # ---------- Validate input files exist ----------
    status = PATHS.validate_inputs()
    print("\nInput file status:")
    for name, ok in status.items():
        path = getattr(PATHS, name)
        print(f"  {'✓' if ok else '✗'} {name:12s} {path}")
    if not all(status.values()):
        print("\n✗ Missing input files. Fix paths in project_config.py and retry.")
        sys.exit(1)
    print()

    # ---------- Load real CSVs ----------
    print("=" * 70)
    print("Real data — train + test CSVs")
    print("=" * 70)
    train_df = pd.read_csv(PATHS.train_csv)
    test_df  = pd.read_csv(PATHS.test_csv)

    print(f"Train: {len(train_df):>5d} rows, "
          f"{train_df['date'].iloc[0]} → {train_df['date'].iloc[-1]}")
    print(f"Test:  {len(test_df):>5d} rows, "
          f"{test_df['date'].iloc[0]} → {test_df['date'].iloc[-1]}")
    print(f"CSV total columns: {len(train_df.columns)}")

    # ---------- Process via env's loader ----------
    sys.path.insert(0, str(PROJECT_ROOT / "env"))
    from portfolio_hrl_env_fixed import process_raw_df, load_synthetic_pool

    feats_train, rets_train, prices_train = process_raw_df(train_df)
    feats_test,  rets_test,  prices_test  = process_raw_df(test_df)

    print(f"\nAfter process_raw_df (date and *_close dropped from features):")
    print(f"  Train features: {feats_train.shape}    (expected: ({len(train_df)}, 313))")
    print(f"  Test features:  {feats_test.shape}     (expected: ({len(test_df)}, 313))")
    print(f"  Train returns:  {rets_train.shape}     (expected: ({len(train_df)}, 4))")
    print(f"  Train prices:   {prices_train.shape}   (held aside, not features)")

    # Hard assertions — fail fast if shapes are wrong
    assert feats_train.shape[1] == 313, f"Expected 313 features, got {feats_train.shape[1]}"
    assert feats_test.shape[1]  == 313
    assert rets_train.shape[1]  == 4
    assert prices_train.shape[1] == 4

    # ---------- NaN / Inf check ----------
    n_nan_train = int(np.isnan(feats_train).sum())
    n_inf_train = int(np.isinf(feats_train).sum())
    n_nan_test  = int(np.isnan(feats_test).sum())
    n_inf_test  = int(np.isinf(feats_test).sum())
    print(f"\nNaN/Inf check:")
    print(f"  Train: NaN={n_nan_train}, Inf={n_inf_train}")
    print(f"  Test:  NaN={n_nan_test},  Inf={n_inf_test}")
    assert n_nan_train == 0 and n_inf_train == 0, "Train features have NaN/Inf"
    assert n_nan_test  == 0 and n_inf_test  == 0, "Test features have NaN/Inf"

    # ---------- Return ranges ----------
    print(f"\nReturn ranges per asset (train, skipping row 0):")
    asset_names = ["NVDA", "AMD", "SMH", "TLT"]
    for i, name in enumerate(asset_names):
        r = rets_train[1:, i]
        print(f"  {name:5s} min={r.min():+.4f}  max={r.max():+.4f}  "
              f"mean={r.mean():+.5f}  std={r.std():.4f}")

    # ---------- Synthetic pool ----------
    print()
    print("=" * 70)
    print("Synthetic pool")
    print("=" * 70)
    pool = load_synthetic_pool(PATHS.synth_pool, drop_close_features=True)
    print(f"  Pool features: {pool['features'].shape}    "
          f"(expected: (2000, 384, 313))")
    print(f"  Pool returns:  {pool['returns'].shape}     "
          f"(expected: (2000, 384, 4))")
    if pool["prices"] is not None:
        print(f"  Pool prices:   {pool['prices'].shape}     "
              f"(extracted from close columns)")

    assert pool["features"].shape[2] == 313, (
        f"Pool feature dim {pool['features'].shape[2]} != 313 — "
        "closes were not dropped properly"
    )
    assert pool["features"].shape[2] == feats_train.shape[1], (
        "Pool and real feature dims must match for the agent to be transferable"
    )
    print("  Pool feature dim matches real data ✓")

    n_nan_pool = int(np.isnan(pool["features"]).sum())
    print(f"  Pool NaN cells: {n_nan_pool} / {pool['features'].size}")
    assert n_nan_pool == 0, "Synthetic pool features have NaN"

    # ---------- DTB3 sanity ----------
    print()
    print("=" * 70)
    print("DTB3 (T-Bill rates) — for cash returns in env (not yet wired in)")
    print("=" * 70)
    dtb3 = pd.read_csv(PATHS.dtb3_csv)
    print(f"  DTB3 rows: {len(dtb3)}")
    print(f"  DTB3 columns: {list(dtb3.columns)}")
    print(f"  Date range: {dtb3['DATE'].iloc[0]} → {dtb3['DATE'].iloc[-1]}")
    print(f"  Rate range: {dtb3['rate'].min():.2f}% → {dtb3['rate'].max():.2f}%")
    print(f"  Daily fraction (CPct_Chg) range: "
          f"{dtb3['CPct_Chg'].min():.2e} → {dtb3['CPct_Chg'].max():.2e}")
    
    # Coverage check: does DTB3 cover the train + test date range?
    dtb3_dates = pd.to_datetime(dtb3["DATE"])
    train_first = pd.to_datetime(train_df["date"].iloc[0])
    test_last   = pd.to_datetime(test_df["date"].iloc[-1])
    dtb3_covers = (dtb3_dates.min() <= train_first) and (dtb3_dates.max() >= test_last)
    print(f"  Covers {train_first.date()} to {test_last.date()}? "
          f"{'✓' if dtb3_covers else '✗'}")
    assert dtb3_covers, "DTB3 doesn't cover the training/test date range"

    # ---------- All clear ----------
    print()
    print("=" * 70)
    print("✓ Step 1 PASSED — all data is clean and shapes line up")
    print("=" * 70)


if __name__ == "__main__":
    main()
