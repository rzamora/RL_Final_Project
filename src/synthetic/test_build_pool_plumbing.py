"""
Standalone integration test of build_pool's data plumbing.

Mocks out the heavy upstream modules (synthetic_feature_builder, simulate_hybrid_paths)
so we can verify the slicing, trimming, and round-trip logic in isolation
without needing the full project environment.

This catches bugs that py_compile won't:
    - Wrong axis slicing on companion arrays (returns/prices/volumes/regimes)
    - Save/load round-trip preserving dtypes and shapes
    - Per-ticker Kronos split correctness
    - Sampler returning correctly-shaped batches
"""

import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

# -----------------------------------------------------------------------------
# Mock the heavy modules BEFORE importing build_synthetic_pool_v2.
#
# Strategy: install fake modules in sys.modules so the imports inside
# build_synthetic_pool_v2 resolve to our stubs.
# -----------------------------------------------------------------------------

# Mock synthetic_feature_builder
sfb = types.ModuleType("synthetic_feature_builder")

KRONOS_COLUMNS_MOCK = [
    "kronos_close_d5", "kronos_pcterr_d1", "kronos_hit_d1",
    "kronos_error_stab", "kronos_conf", "kronos_surprise",
    "kronos_regime_0", "kronos_regime_1", "kronos_regime_2",
    "kronos_regime_3", "kronos_regime_4", "kronos_regime_5",
    "kronos_regime_6", "kronos_regime_7", "kronos_regime_8",
    "Kronos_Slope", "Kronos_Convexity", "Kronos_TermStructureCorr",
    "kronos_band_skew_mean",
]
REGIME_PROB_COLS_MOCK = [
    "regime_prob_Bull", "regime_prob_Bear",
    "regime_prob_SevereBear", "regime_prob_Crisis",
]


def mock_build_kronos_block_for_path(synth_regime_path, real_dfs_by_ticker,
                                      real_regimes_by_ticker, kronos_columns,
                                      asset_tickers, top_k, sample_temperature, seed):
    n_steps = len(synth_regime_path)
    cols = []
    for ticker in asset_tickers:
        for c in kronos_columns:
            cols.append(f"{ticker}_{c}")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, size=(n_steps, len(cols))), columns=cols)


def mock_build_synthetic_features(returns_path, volumes_path, prices_path,
                                   regime_path, asset_tickers, equity_assets,
                                   tlt_asset, kronos_block_df,
                                   dirichlet_concentration, dirichlet_noise_floor,
                                   trim_warmup, start_date, seed, verbose):
    """
    Returns a DataFrame with the expected schema shape: a 'date' column,
    ticker-prefixed feature columns, regime probability columns, and
    Kronos columns. n_clean_steps = len - trim_warmup.
    """
    n_steps = len(regime_path)
    n_clean = n_steps - trim_warmup

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_clean)
    out = pd.DataFrame({"date": dates})

    # A few non-Kronos per-asset features (just enough to look real)
    for ticker in asset_tickers:
        for col in ["close", "vol_10", "RSI_norm_14"]:
            out[f"{ticker}_{col}"] = rng.normal(0, 1, n_clean).astype(np.float32)

    # Cross-asset corr columns
    for pair in ["NVDA_AMD", "NVDA_SMH", "NVDA_TLT"]:
        out[f"corr_{pair}_20"] = rng.uniform(-1, 1, n_clean).astype(np.float32)

    # Regime probabilities
    alpha = np.full((n_clean, 4), dirichlet_noise_floor)
    rng2 = np.random.default_rng(seed + 1)
    g = rng2.gamma(shape=alpha, scale=1.0)
    probs = g / g.sum(axis=1, keepdims=True)
    for i, c in enumerate(REGIME_PROB_COLS_MOCK):
        out[c] = probs[:, i].astype(np.float32)

    # Kronos columns (trimmed from kronos_block_df)
    if kronos_block_df is not None:
        kb_trim = kronos_block_df.iloc[trim_warmup:].reset_index(drop=True)
        for col in kb_trim.columns:
            out[col] = kb_trim[col].values

    return out


def mock_validate_schema(df, ref_path):
    return True


def mock_reorder_to_reference(df, ref_path):
    return df


sfb.KRONOS_COLUMNS = KRONOS_COLUMNS_MOCK
sfb.REGIME_PROB_COLS = REGIME_PROB_COLS_MOCK
sfb.build_kronos_block_for_path = mock_build_kronos_block_for_path
sfb.build_synthetic_features = mock_build_synthetic_features
sfb.validate_schema = mock_validate_schema
sfb.reorder_to_reference = mock_reorder_to_reference
sys.modules["synthetic_feature_builder"] = sfb

# Mock regime_dcc_garch_copula_V1
rdgc = types.ModuleType("regime_dcc_garch_copula_V1")


def mock_simulate_hybrid_paths(regime_models, trans_mat, assets, initial_regime,
                                n_steps, n_paths, return_volumes, return_prices,
                                stress_bias, seed, start_price):
    rng = np.random.default_rng(seed)
    N = len(assets)
    all_returns = rng.normal(0.05, 1.5, size=(n_paths, n_steps, N)).astype(np.float64)
    all_regimes = rng.integers(0, 4, size=(n_paths, n_steps)).astype(np.int64)
    all_volumes = np.exp(
        rng.normal(17, 0.5, size=(n_paths, n_steps, N))
    ).astype(np.float64)
    all_prices = np.zeros((n_paths, n_steps + 1, N))
    all_prices[:, 0, :] = start_price
    for t in range(n_steps):
        all_prices[:, t + 1, :] = all_prices[:, t, :] * (1 + all_returns[:, t, :] / 100.0)
    return all_returns, all_regimes, all_volumes, all_prices


rdgc.simulate_hybrid_paths = mock_simulate_hybrid_paths
sys.modules["regime_dcc_garch_copula_V1"] = rdgc

# Now we can import build_synthetic_pool_v2 cleanly
from build_synthetic_pool_v2 import (
    build_pool,
    load_pool,
    verify_round_trip,
    PoolSampler,
    split_merged_into_per_ticker_kronos,
    build_per_ticker_regime_dict,
    load_fitted_pickle,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
TMPDIR = Path("/tmp/pool_test_v2")
TMPDIR.mkdir(exist_ok=True)

ASSET_TICKERS = ("NVDA", "AMD", "SMH", "TLT")


def make_fake_pickle():
    """Create a fake fitted pickle for testing."""
    import pickle
    n_real = 2000
    rng = np.random.default_rng(0)
    params = {
        "regime_models": [{"placeholder": k} for k in range(4)],
        "trans_mat": np.full((4, 4), 0.25),
        "regime_seq": rng.integers(0, 4, size=n_real),
        "regime_probs": rng.dirichlet([1, 1, 1, 1], size=n_real),
        "training_dates": pd.bdate_range("2010-01-01", periods=n_real),
    }
    pkl_path = TMPDIR / "fake_fitted.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(params, f)
    return pkl_path, n_real


def make_fake_merged_csv(n_real):
    """Create a fake merged train CSV with the right Kronos column structure."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-01", periods=n_real)
    df = pd.DataFrame({"date": dates})

    # A few non-Kronos columns (simulating per-asset feature blocks)
    for ticker in ASSET_TICKERS:
        for col in ["close", "vol_10", "RSI_norm_14"]:
            df[f"{ticker}_{col}"] = rng.normal(0, 1, n_real)

    # Cross-asset
    for pair in ["NVDA_AMD", "NVDA_SMH", "NVDA_TLT"]:
        df[f"corr_{pair}_20"] = rng.uniform(-1, 1, n_real)

    # Regime probability cols
    for c in REGIME_PROB_COLS_MOCK:
        df[c] = rng.uniform(0, 1, n_real)

    # Kronos cols (ticker-prefixed)
    for ticker in ASSET_TICKERS:
        for c in KRONOS_COLUMNS_MOCK:
            df[f"{ticker}_{c}"] = rng.normal(0, 0.02, n_real)

    csv_path = TMPDIR / "fake_merged_train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def make_fake_short_merged_csv(n_real, n_train):
    """
    Make a CSV that has FEWER rows than the fitted pickle's regime_seq —
    simulates a train/test split where the pickle was fit on full history
    but the CSV is the train slice. Returns CSV path + the dates kept.
    """
    rng = np.random.default_rng(2)
    full_dates = pd.bdate_range("2010-01-01", periods=n_real)
    train_dates = full_dates[:n_train]  # take first n_train dates as 'train'

    df = pd.DataFrame({"date": train_dates})
    for ticker in ASSET_TICKERS:
        for col in ["close", "vol_10", "RSI_norm_14"]:
            df[f"{ticker}_{col}"] = rng.normal(0, 1, n_train)
    for pair in ["NVDA_AMD", "NVDA_SMH", "NVDA_TLT"]:
        df[f"corr_{pair}_20"] = rng.uniform(-1, 1, n_train)
    for c in REGIME_PROB_COLS_MOCK:
        df[c] = rng.uniform(0, 1, n_train)
    for ticker in ASSET_TICKERS:
        for c in KRONOS_COLUMNS_MOCK:
            df[f"{ticker}_{c}"] = rng.normal(0, 0.02, n_train)

    csv_path = TMPDIR / "fake_merged_train_SHORT.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, train_dates


# -----------------------------------------------------------------------------
# Test suite
# -----------------------------------------------------------------------------
def test_split_merged():
    print("\n[test_split_merged]")
    pkl_path, n_real = make_fake_pickle()
    csv_path = make_fake_merged_csv(n_real)
    merged = pd.read_csv(csv_path)
    real_dfs = split_merged_into_per_ticker_kronos(merged, ASSET_TICKERS, KRONOS_COLUMNS_MOCK)
    for ticker in ASSET_TICKERS:
        assert ticker in real_dfs, f"missing ticker {ticker}"
        df = real_dfs[ticker]
        assert "date" in df.columns
        for c in KRONOS_COLUMNS_MOCK:
            assert c in df.columns, f"{ticker} missing unprefixed {c}"
            # The values should match the prefixed source column
            assert np.array_equal(df[c].values, merged[f"{ticker}_{c}"].values), \
                f"{ticker} {c} values don't match source"
    print(f"  OK — split into {len(real_dfs)} tickers, "
          f"each with {len(real_dfs[ASSET_TICKERS[0]].columns)} cols")


def test_per_ticker_regime_dict():
    print("\n[test_per_ticker_regime_dict]")
    seq = np.array([0, 1, 2, 3, 0, 1])
    d = build_per_ticker_regime_dict(seq, ASSET_TICKERS)
    for ticker in ASSET_TICKERS:
        assert ticker in d
        assert np.array_equal(d[ticker], seq)
    print(f"  OK — {len(d)} tickers, all sharing the same {len(seq)}-day regime seq")


def test_load_fitted_pickle_validation():
    print("\n[test_load_fitted_pickle_validation]")
    pkl_path, _ = make_fake_pickle()
    params = load_fitted_pickle(pkl_path)
    for k in ["regime_models", "trans_mat", "regime_seq", "training_dates"]:
        assert k in params, f"missing key {k}"
    print(f"  OK — loaded keys: {list(params.keys())}")

    # Test missing-key detection
    import pickle
    bad_path = TMPDIR / "bad_pickle.pkl"
    with open(bad_path, "wb") as f:
        pickle.dump({"only_one_key": "value"}, f)
    try:
        load_fitted_pickle(bad_path)
        raise AssertionError("Should have raised KeyError")
    except KeyError as e:
        print(f"  OK — correctly raised on incomplete pickle: {e}")


def test_build_pool_small():
    print("\n[test_build_pool_small]")
    pkl_path, n_real = make_fake_pickle()
    csv_path = make_fake_merged_csv(n_real)

    n_paths, n_steps, trim_warmup = 5, 200, 50
    n_clean = n_steps - trim_warmup

    pool = build_pool(
        fitted_pickle_path=pkl_path,
        merged_train_csv_path=csv_path,
        n_paths=n_paths,
        n_steps=n_steps,
        trim_warmup=trim_warmup,
        seed=42,
        asset_tickers=ASSET_TICKERS,
        save_to=None,
        validate_against_reference=False,  # mocks don't enforce schema
        reorder_columns_to_reference=False,
        verbose=False,
    )

    # Shape checks
    assert pool["features"].shape[0] == n_paths
    assert pool["features"].shape[1] == n_clean, \
        f"expected {n_clean} clean rows, got {pool['features'].shape[1]}"
    assert pool["returns"].shape == (n_paths, n_clean, len(ASSET_TICKERS))
    assert pool["volumes"].shape == (n_paths, n_clean, len(ASSET_TICKERS))
    # Prices kept at trim_warmup: → length n_steps + 1 - trim_warmup = n_clean + 1
    assert pool["prices"].shape == (n_paths, n_steps + 1 - trim_warmup, len(ASSET_TICKERS)), \
        f"prices shape {pool['prices'].shape}, expected {(n_paths, n_steps + 1 - trim_warmup, len(ASSET_TICKERS))}"
    assert pool["regimes"].shape == (n_paths, n_clean)

    # Dtype checks
    assert pool["features"].dtype == np.float32
    assert pool["returns"].dtype == np.float32
    assert pool["regimes"].dtype == np.int8

    # feature_names cardinality matches n_features
    assert len(pool["feature_names"]) == pool["features"].shape[2]

    # Path diversity — paths must differ
    means = pool["features"][:, :, 0].mean(axis=1)
    assert means.std() > 0, "Paths look identical — RNG diversity broken"

    print(f"  OK — features {pool['features'].shape}, returns {pool['returns'].shape}")
    print(f"       prices {pool['prices'].shape}, regimes {pool['regimes'].shape}")
    print(f"       n_features = {len(pool['feature_names'])}, "
          f"path means std = {means.std():.3f}")


def test_round_trip():
    print("\n[test_round_trip]")
    pkl_path, n_real = make_fake_pickle()
    csv_path = make_fake_merged_csv(n_real)

    pool = build_pool(
        fitted_pickle_path=pkl_path,
        merged_train_csv_path=csv_path,
        n_paths=4,
        n_steps=200,
        trim_warmup=50,
        seed=7,
        asset_tickers=ASSET_TICKERS,
        save_to=None,
        validate_against_reference=False,
        reorder_columns_to_reference=False,
        verbose=False,
    )
    save_path = TMPDIR / "round_trip_pool.npz"
    reloaded = verify_round_trip(pool, save_path, verbose=True)

    # Metadata round-trip
    assert reloaded["metadata"]["seed"] == 7
    assert reloaded["metadata"]["n_paths"] == 4
    assert reloaded["metadata"]["n_clean_steps"] == 150
    print("  OK — metadata round-trips correctly")


def test_sampler():
    print("\n[test_sampler]")
    pkl_path, n_real = make_fake_pickle()
    csv_path = make_fake_merged_csv(n_real)
    pool = build_pool(
        fitted_pickle_path=pkl_path, merged_train_csv_path=csv_path,
        n_paths=10, n_steps=200, trim_warmup=50, seed=1,
        asset_tickers=ASSET_TICKERS, save_to=None,
        validate_against_reference=False, reorder_columns_to_reference=False,
        verbose=False,
    )
    sampler = PoolSampler(pool)
    rng = np.random.default_rng(0)

    # Small batch
    batch = sampler.sample(batch_size=3, rng=rng, replace=False)
    assert batch["features"].shape[0] == 3
    assert batch["features"].shape[1:] == pool["features"].shape[1:]
    assert len(batch["path_idx"]) == 3
    assert len(set(batch["path_idx"].tolist())) == 3, "replace=False should give unique idx"
    print(f"  OK — sampled batch shapes: {[(k, v.shape if hasattr(v, 'shape') else v) for k, v in batch.items()]}")

    # Larger than pool with replace=True
    batch2 = sampler.sample(batch_size=20, rng=rng, replace=True)
    assert batch2["features"].shape[0] == 20
    print(f"  OK — replace=True sampling beyond pool size works")


def test_trim_alignment():
    """
    Critical: the per-step alignment between features[t] and returns[t] must
    be preserved through the trim. Verify this with a known-input check.
    """
    print("\n[test_trim_alignment]")
    pkl_path, n_real = make_fake_pickle()
    csv_path = make_fake_merged_csv(n_real)

    n_paths, n_steps, trim_warmup = 2, 150, 30
    pool = build_pool(
        fitted_pickle_path=pkl_path, merged_train_csv_path=csv_path,
        n_paths=n_paths, n_steps=n_steps, trim_warmup=trim_warmup, seed=99,
        asset_tickers=ASSET_TICKERS, save_to=None,
        validate_against_reference=False, reorder_columns_to_reference=False,
        verbose=False,
    )
    n_clean = n_steps - trim_warmup
    assert pool["features"].shape[1] == n_clean
    assert pool["returns"].shape[1] == n_clean
    assert pool["regimes"].shape[1] == n_clean
    assert pool["volumes"].shape[1] == n_clean
    # prices is +1 in the time dim (close at start + close after each return)
    assert pool["prices"].shape[1] == n_clean + 1
    print(f"  OK — features/returns/volumes/regimes all = {n_clean}, "
          f"prices = {n_clean + 1}")


def test_train_test_split_alignment():
    """
    Pickle has 4757 'real' dates; CSV is the train slice with 4579 dates.
    The runner used to assert equality and crash. The new logic should align
    by date.
    """
    print("\n[test_train_test_split_alignment]")
    pkl_path, n_real = make_fake_pickle()  # n_real = 2000
    n_train = 1500
    csv_path, train_dates = make_fake_short_merged_csv(n_real, n_train)

    pool = build_pool(
        fitted_pickle_path=pkl_path,
        merged_train_csv_path=csv_path,
        n_paths=3, n_steps=200, trim_warmup=50, seed=11,
        asset_tickers=ASSET_TICKERS, save_to=None,
        validate_against_reference=False, reorder_columns_to_reference=False,
        verbose=False,
    )
    # Should have built 3 paths despite the length mismatch
    assert pool["features"].shape[0] == 3
    assert pool["features"].shape[1] == 150
    print(f"  OK — pickle had {n_real} dates, CSV had {n_train}, "
          f"build succeeded with date-based alignment")


def test_csv_with_unknown_dates_raises():
    """
    If the CSV contains dates that aren't in the pickle's training_dates,
    we should fail loudly with a useful error message — not silently mismatch.
    """
    print("\n[test_csv_with_unknown_dates_raises]")
    pkl_path, n_real = make_fake_pickle()  # dates = bdate_range starting 2010-01-01
    # Make a CSV with dates that DON'T overlap the pickle range
    rng = np.random.default_rng(3)
    bad_dates = pd.bdate_range("2050-01-01", periods=100)  # far future, not in pickle
    df = pd.DataFrame({"date": bad_dates})
    for ticker in ASSET_TICKERS:
        for col in ["close", "vol_10", "RSI_norm_14"]:
            df[f"{ticker}_{col}"] = rng.normal(0, 1, 100)
    for pair in ["NVDA_AMD", "NVDA_SMH", "NVDA_TLT"]:
        df[f"corr_{pair}_20"] = rng.uniform(-1, 1, 100)
    for c in REGIME_PROB_COLS_MOCK:
        df[c] = rng.uniform(0, 1, 100)
    for ticker in ASSET_TICKERS:
        for c in KRONOS_COLUMNS_MOCK:
            df[f"{ticker}_{c}"] = rng.normal(0, 0.02, 100)
    bad_csv = TMPDIR / "bad_dates.csv"
    df.to_csv(bad_csv, index=False)

    try:
        build_pool(
            fitted_pickle_path=pkl_path, merged_train_csv_path=bad_csv,
            n_paths=2, n_steps=200, trim_warmup=50, seed=1,
            asset_tickers=ASSET_TICKERS, save_to=None,
            validate_against_reference=False,
            reorder_columns_to_reference=False, verbose=False,
        )
        raise AssertionError("Should have raised KeyError on unknown dates")
    except KeyError as e:
        msg = str(e)
        assert "not in the fitted pickle" in msg, \
            f"Error message should mention pickle mismatch, got: {msg}"
        print(f"  OK — raised informative KeyError on unknown dates")


def test_round_trip_with_nan_features():
    """
    Build a pool, manually inject NaN into the features tensor, and verify
    the round-trip check still passes. NaNs are legitimate (wavelet/technical
    indicators produce them), and np.savez preserves them — but np.array_equal
    needs equal_nan=True to count NaN-in-same-position as equal.
    """
    print("\n[test_round_trip_with_nan_features]")
    pkl_path, n_real = make_fake_pickle()
    csv_path = make_fake_merged_csv(n_real)

    pool = build_pool(
        fitted_pickle_path=pkl_path, merged_train_csv_path=csv_path,
        n_paths=3, n_steps=200, trim_warmup=50, seed=99,
        asset_tickers=ASSET_TICKERS, save_to=None,
        validate_against_reference=False, reorder_columns_to_reference=False,
        verbose=False,
    )

    # Inject NaN into a few cells of features (simulates wavelet warmup edge cases)
    features = pool["features"]
    features[0, 0, 0] = np.nan
    features[1, 5, 3] = np.nan
    features[2, -1, 7] = np.nan
    pool["features"] = features
    n_nan_inserted = 3

    save_path = TMPDIR / "round_trip_with_nan.npz"
    reloaded = verify_round_trip(pool, save_path, verbose=False)

    n_nan_reloaded = int(np.isnan(reloaded["features"]).sum())
    assert n_nan_reloaded == n_nan_inserted, \
        f"Expected {n_nan_inserted} NaNs after reload, got {n_nan_reloaded}"
    print(f"  OK — round-trip preserved {n_nan_inserted} NaNs in features tensor")


# -----------------------------------------------------------------------------
# Run all
# -----------------------------------------------------------------------------
def main():
    tests = [
        test_split_merged,
        test_per_ticker_regime_dict,
        test_load_fitted_pickle_validation,
        test_build_pool_small,
        test_round_trip,
        test_sampler,
        test_trim_alignment,
        test_train_test_split_alignment,
        test_csv_with_unknown_dates_raises,
        test_round_trip_with_nan_features,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
