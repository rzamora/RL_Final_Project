"""
Synthetic feature builder for RL training.

Takes the raw output of the regime-DCC-GARCH-copula generator and produces
a feature DataFrame with the same schema as RL_Final_Merged_train.csv,
minus Kronos columns (those are filled in by KronosAligner).

Pipeline per synthetic path:
    1. Build per-asset DataFrames (date, close, Volume) — close-only
    2. Run Technicals_TimeSeries: volume() + ml_safe_technicals(), lowCorrelation=True
    3. Run Technicals_Wavelets at windows 90 and 128, then compute_cross_scale_features
    4. Trim per-asset DFs to the canonical 68-column per-asset schema
    5. add_correlation_features over 4-asset return panel
    6. merge_asset_features into wide format with ticker prefixes
    7. Append Dirichlet-smoothed regime probability columns
    8. (Externally) attach Kronos block from KronosAligner
    9. Trim warmup rows (first `trim_warmup` rows lost to wavelet window)

Output column order matches RL_Final_Merged_train.csv exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path

# These imports come from your existing modules.
# They must be importable from the working directory.
from IB_Data import Technicals_TimeSeries
from IB_Data import Technicals_Wavelets
from cross_asset_correlations import add_correlation_features
from merge_asset_features_into_one import merge_asset_features


# =============================================================================
# Canonical schema definitions
# =============================================================================
# The 68 per-asset feature columns (in order, excluding date and TICKER).
# This list is the contract — synthetic per-asset DFs must end up with exactly
# these columns before merging.
PER_ASSET_FEATURES = [
    "close",
    "V_mad_z20", "V_chg1_%",
    "CPct_Chg1", "CPct_Chg1Lg1", "CPct_Chg1Lg2", "CPct_Chg1Lg3",
    "CPct_Chg1Lg4", "CPct_Chg1Lg5",
    "Fidx_ML", "rOBV", "Fidx_ML_10", "Fidx_ML_50",
    "mChM_ML_10", "mChM_ML_50",
    "AD_ML", "ChOsc_ML", "MFI_ML_21",
    "vol_10", "vol_60", "vol_ratio_10_60",
    "MACD_norm", "MACD_hist_norm",
    "RSI_norm_5", "RSI_norm_14", "RSI_norm_20", "RSI_norm_50",
    "PDI_norm_14", "MDI_norm_14", "ADX_norm_14",
    # Kronos block (16 cols) — filled later by KronosAligner
    "kronos_close_d5", "kronos_pcterr_d1", "kronos_hit_d1",
    "kronos_error_stab", "kronos_conf", "kronos_surprise",
    "kronos_regime_0", "kronos_regime_1", "kronos_regime_2",
    "kronos_regime_3", "kronos_regime_4", "kronos_regime_5",
    "kronos_regime_6", "kronos_regime_7", "kronos_regime_8",
    "Kronos_Slope", "Kronos_Convexity", "Kronos_TermStructureCorr",
    "kronos_band_skew_mean",
    # Wavelet block (18 cols)
    "wv_entropy128_L0", "wv_energy128_L4", "wv_entropy128_L4",
    "wv_trend_slope_128", "wv_noise_ratio_128", "wv_mid_energy_ratio_128",
    "wv_entropy90_L0", "wv_entropy90_L2", "wv_mid_energy_ratio_90",
    "slope_momentum_90_128", "regime_agree_90_128",
    "wv_regime_128_4_Bear", "wv_regime_128_4_Bull",
    "wv_regime_128_4_Sideways", "wv_regime_128_4_Transition",
    "wv_regime_90_4_Bear", "wv_regime_90_4_Bull",
    "wv_regime_90_4_Sideways", "wv_regime_90_4_Transition",
]

# Wavelet columns we compute then drop (these are the "non-keep" wavelet outputs
# that compute_cross_scale_features's lowCorrelation=True drops anyway, but we
# also need to drop intermediate wavelet outputs that don't survive)
KRONOS_COLUMNS = [c for c in PER_ASSET_FEATURES if c.startswith("kronos_") or c.startswith("Kronos_")]

# Wavelet regime one-hot columns — these may legitimately be absent from a
# given path's output if the regime classifier never assigned that label
# (e.g., a calm path that never enters "Transition"). We fill missing ones
# with 0 rather than raising.
WAVELET_REGIME_ONEHOT_COLS = [
    "wv_regime_128_4_Bear", "wv_regime_128_4_Bull",
    "wv_regime_128_4_Sideways", "wv_regime_128_4_Transition",
    "wv_regime_90_4_Bear", "wv_regime_90_4_Bull",
    "wv_regime_90_4_Sideways", "wv_regime_90_4_Transition",
]

REGIME_PROB_COLS = ["regime_prob_Bull", "regime_prob_Bear",
                    "regime_prob_SevereBear", "regime_prob_Crisis"]

# Regime label mapping confirmed by user
REGIME_NAME_BY_INDEX = {0: "Bull", 1: "Bear", 2: "SevereBear", 3: "Crisis"}


# =============================================================================
# Per-asset DataFrame construction
# =============================================================================
def _build_per_asset_dfs(returns_path, volumes_path, prices_path, asset_tickers,
                          start_date="2020-01-01"):
    """
    Build {ticker: df} dict from one synthetic path.

    Each df has: date, close, Volume, TICKER columns.
    No high/low/open — the close-only branches in Technicals_TimeSeries handle this.

    Parameters
    ----------
    returns_path : np.ndarray, shape (n_steps, n_assets)
    volumes_path : np.ndarray, shape (n_steps, n_assets)
    prices_path  : np.ndarray, shape (n_steps+1, n_assets)
        prices[0] is the initial price; prices[1:] correspond to returns[:].
    asset_tickers : tuple[str]
    start_date : str
        Fake business-day index will start here.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    n_steps = returns_path.shape[0]
    n_assets = returns_path.shape[1]
    if len(asset_tickers) != n_assets:
        raise ValueError(f"asset_tickers has {len(asset_tickers)} entries but "
                         f"returns has {n_assets} columns")
    if prices_path.shape[0] != n_steps + 1:
        raise ValueError(f"prices_path should have n_steps+1 rows; "
                         f"got {prices_path.shape[0]} vs n_steps={n_steps}")

    dates = pd.bdate_range(start=start_date, periods=n_steps)

    asset_dfs = {}
    for i, ticker in enumerate(asset_tickers):
        # Use prices[1:] — these are the prices AFTER each return is realized.
        # That's the "close" at each timestep.
        df = pd.DataFrame({
            "date": dates,
            "TICKER": ticker,
            "close": prices_path[1:, i].astype(float),
            "Volume": volumes_path[:, i].astype(float),
        })
        asset_dfs[ticker] = df
    return asset_dfs


# =============================================================================
# Tactical features (volume + technicals)
# =============================================================================
def _run_tactical_features(asset_dfs):
    """
    Run Technicals_TimeSeries.volume() then .ml_safe_technicals() with lowCorrelation=True.
    Modifies and returns the dict.

    Also fills the row-0 NaN in CPct_Chg1 with 0.0 to keep row counts consistent
    downstream — the cross-asset correlation module does .dropna() when building
    its returns panel, which would otherwise produce a panel that's 1 row shorter
    than the per-asset DFs and break the merge step.
    """
    ts = Technicals_TimeSeries(asset_dfs)
    ts.volume(extraInd=True, lowCorrelation=True)
    ts.ml_safe_technicals(
        ma_windows=[10, 20, 60],
        rsi_periods=[5, 14, 20, 50],
        adx_periods=[14],
        stoch_windows=[50],
        macd_params=(12, 26, 9),
        vol_windows=[10, 60],
        clip_val=5,
        lowCorrelation=True,
    )
    # Fill row-0 NaN in CPct_Chg1 (and its lags) so cross-asset panel keeps
    # all rows. A 0 first-day return is consistent with the synthetic
    # generator convention where prices[0] is the starting price.
    for ticker, df in ts.data.items():
        for col in ["CPct_Chg1", "CPct_Chg1Lg1", "CPct_Chg1Lg2",
                    "CPct_Chg1Lg3", "CPct_Chg1Lg4", "CPct_Chg1Lg5"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        ts.data[ticker] = df

    return ts.data


# =============================================================================
# Wavelet features
# =============================================================================
def _run_wavelet_features(asset_dfs, windows=(90, 128), n_levels=4):
    """
    Run Technicals_Wavelets at each window, then compute_cross_scale_features.
    Returns the updated dict.
    """
    wave = Technicals_Wavelets(asset_dfs)
    # Compute at both windows. label_targets() also one-hots the wv_regime
    # categorical column via implicit dependency on compute_cross_scale_features's
    # one_hot_encode_strings call below.
    for ww in windows:
        for tck in wave.tickers:
            wave.compute_Wavelet_features_V2(tck, wavelet_window=ww, n_levels=n_levels)
            wave.label_targets(tck, wavelet_window=ww, n_levels=n_levels)

    # cross-scale: differences and regime agreement; also one-hots the regime cols
    ww1, ww2 = windows  # (90, 128) per the schema
    wave.compute_cross_scale_features(ww1, ww2, nl1=n_levels, nl2=n_levels,
                                       lowCorrelation=True)
    return wave.data


# =============================================================================
# Per-asset trimming to canonical schema
# =============================================================================
def _trim_per_asset_to_schema(asset_dfs, fill_kronos_nan=True):
    """
    Reduce each per-asset DataFrame to exactly the 68 PER_ASSET_FEATURES columns
    (plus date and TICKER). Kronos columns are filled with NaN as placeholders
    (will be overwritten by KronosAligner). Wavelet regime one-hot columns
    that are absent because the classifier never assigned that label on this
    path are filled with 0.

    Raises if any required non-Kronos, non-wavelet-regime column is missing.
    """
    trimmed = {}
    for ticker, df in asset_dfs.items():
        out = pd.DataFrame({"date": pd.to_datetime(df["date"]), "TICKER": ticker})

        missing_required = []
        for col in PER_ASSET_FEATURES:
            if col in df.columns:
                out[col] = df[col].values
            elif col in KRONOS_COLUMNS and fill_kronos_nan:
                out[col] = np.nan  # placeholder; KronosAligner will fill
            elif col in WAVELET_REGIME_ONEHOT_COLS:
                # Classifier never assigned this regime on this path — that's fine,
                # the one-hot just stays 0 for every row.
                out[col] = 0
            else:
                missing_required.append(col)

        if missing_required:
            available = sorted(df.columns.tolist())
            raise KeyError(
                f"[{ticker}] Missing required columns after feature build: "
                f"{missing_required}\n"
                f"Available columns: {available}"
            )

        trimmed[ticker] = out
    return trimmed


# =============================================================================
# Cross-asset correlations + merge
# =============================================================================
def _add_cross_asset_and_merge(asset_dfs, asset_tickers, equity_assets, tlt_asset):
    """
    Compute cross-asset correlation features, then merge per-asset + cross-asset
    blocks into one wide DataFrame.
    """
    # add_correlation_features returns the cross-asset DF (not enriched per-asset)
    cross_df = add_correlation_features(
        asset_dfs,
        return_col="CPct_Chg1",
        date_col="date",
        windows=(20, 60),
        zscore_window=(30, 120),
        equity_assets=list(equity_assets),
        tlt_asset=tlt_asset,
        verbose=False,
    )
    # cross_df already contains a 'date' column (it was reset_index'd).

    # Build the merge dict: per-asset DFs keyed by ticker, plus 'COR' for cross-asset
    merge_input = {**asset_dfs, "COR": cross_df}
    merged = merge_asset_features(
        merge_input,
        asset_tickers=tuple(asset_tickers),
        cross_asset_ticker="COR",
        date_col="date",
        drop_cols=("TICKER", "index"),
        save_to=None,
        verbose=False,
    )
    return merged


# =============================================================================
# Dirichlet-smoothed regime probabilities
# =============================================================================
def _add_regime_probs(merged_df, regime_path, concentration=50.0,
                      noise_floor=0.5, rng=None):
    """
    Append Dirichlet-smoothed regime probability columns.

    For each timestep with hard regime label r:
        alpha = noise_floor + concentration * onehot(r)
        p ~ Dirichlet(alpha)

    Higher `concentration` → probabilities concentrated near the one-hot.
    `noise_floor` ensures all four classes get some mass even when
    concentration is large, so log-probs don't explode.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged wide DataFrame; will have regime_prob_* columns appended.
    regime_path : np.ndarray, shape (n_steps,) of int
        Hard regime labels, must be in {0, 1, 2, 3}.
    concentration : float
        Dirichlet concentration on the true class.
    noise_floor : float
        Baseline alpha for non-true classes.
    rng : np.random.Generator or None

    Returns
    -------
    merged_df : pd.DataFrame
        Same DataFrame with 4 new columns appended (in canonical order).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(regime_path)
    if len(merged_df) != n:
        raise ValueError(f"regime_path length {n} doesn't match merged_df rows {len(merged_df)}")

    n_regimes = 4
    alpha = np.full((n, n_regimes), noise_floor, dtype=float)
    rows = np.arange(n)
    alpha[rows, regime_path.astype(int)] += concentration

    # Vectorized Dirichlet draw via Gamma normalization
    g = rng.gamma(shape=alpha, scale=1.0)
    probs = g / g.sum(axis=1, keepdims=True)

    for i, col in enumerate(REGIME_PROB_COLS):
        merged_df[col] = probs[:, i].astype(np.float32)

    return merged_df


# =============================================================================
# Kronos attachment
# =============================================================================
def _attach_kronos_block(merged_df, kronos_block_df, asset_tickers):
    """
    Overwrite the Kronos placeholder columns in merged_df with values from
    kronos_block_df.

    kronos_block_df must contain ticker-prefixed Kronos columns like
    'NVDA_kronos_close_d5', 'AMD_Kronos_Slope', etc., one row per timestep.
    """
    if len(kronos_block_df) != len(merged_df):
        raise ValueError(
            f"kronos_block has {len(kronos_block_df)} rows but merged_df has {len(merged_df)}"
        )

    expected_kronos_cols = []
    for ticker in asset_tickers:
        for c in KRONOS_COLUMNS:
            expected_kronos_cols.append(f"{ticker}_{c}")

    missing_in_block = [c for c in expected_kronos_cols if c not in kronos_block_df.columns]
    if missing_in_block:
        raise KeyError(f"kronos_block missing columns: {missing_in_block[:5]}...")

    for col in expected_kronos_cols:
        merged_df[col] = kronos_block_df[col].values

    return merged_df


# =============================================================================
# Master entry point
# =============================================================================
def build_synthetic_features(
    returns_path,
    volumes_path,
    prices_path,
    regime_path,
    asset_tickers=("NVDA", "AMD", "SMH", "TLT"),
    equity_assets=("NVDA", "AMD", "SMH"),
    tlt_asset="TLT",
    kronos_block_df=None,
    dirichlet_concentration=50.0,
    dirichlet_noise_floor=0.5,
    trim_warmup=128,
    start_date="2020-01-01",
    seed=None,
    verbose=False,
):
    """
    Build a feature DataFrame for one synthetic path matching the schema of
    RL_Final_Merged_train.csv.

    Parameters
    ----------
    returns_path : np.ndarray, shape (n_steps, 4)
        Daily returns in PERCENT (matches the generator's convention).
    volumes_path : np.ndarray, shape (n_steps, 4)
    prices_path : np.ndarray, shape (n_steps+1, 4)
    regime_path : np.ndarray, shape (n_steps,) of int in {0,1,2,3}
    asset_tickers : tuple[str], length 4
    kronos_block_df : pd.DataFrame, optional
        Pre-aligned Kronos features (output of KronosAligner.assign for this
        path, then ticker-prefixed and merged). Length must equal n_steps.
        If None, Kronos columns will be left as NaN.
    dirichlet_concentration, dirichlet_noise_floor : float
        Controls regime_prob_* spread. With concentration=50, noise=0.5, the
        true regime gets ~99% of mass on average with realistic small leakage
        to other classes.
    trim_warmup : int
        Number of rows to drop from the start. Should be >= max wavelet window
        (default 128) to avoid NaN rows from wavelet warmup.
    seed : int, optional
        For Dirichlet sampling reproducibility.
    verbose : bool

    Returns
    -------
    df : pd.DataFrame
        n_steps - trim_warmup rows × 318 columns.
    """
    if verbose:
        print(f"[build] returns: {returns_path.shape}, volumes: {volumes_path.shape}, "
              f"prices: {prices_path.shape}, regimes: {regime_path.shape}")

    # 1. Per-asset DataFrames
    asset_dfs = _build_per_asset_dfs(
        returns_path, volumes_path, prices_path, asset_tickers,
        start_date=start_date,
    )
    if verbose:
        print(f"[build] Built per-asset DFs: {list(asset_dfs.keys())}, "
              f"each {len(next(iter(asset_dfs.values())))} rows")

    # 2. Tactical features
    asset_dfs = _run_tactical_features(asset_dfs)
    if verbose:
        sample_cols = list(next(iter(asset_dfs.values())).columns)
        print(f"[build] After tactical: {len(sample_cols)} cols, e.g. "
              f"{sample_cols[:8]}...")

    # 3. Wavelet features
    asset_dfs = _run_wavelet_features(asset_dfs, windows=(90, 128), n_levels=4)
    if verbose:
        sample_cols = list(next(iter(asset_dfs.values())).columns)
        wv_cols = [c for c in sample_cols if c.startswith("wv_") or "regime" in c]
        print(f"[build] After wavelets: {len(sample_cols)} cols total, "
              f"{len(wv_cols)} wavelet/regime cols")

    # 4. Trim per-asset DFs to canonical schema (Kronos cols filled with NaN)
    asset_dfs = _trim_per_asset_to_schema(asset_dfs, fill_kronos_nan=True)
    if verbose:
        print(f"[build] Trimmed each asset DF to "
              f"{len(next(iter(asset_dfs.values())).columns)} cols")

    # 5+6. Cross-asset correlations + merge
    merged = _add_cross_asset_and_merge(
        asset_dfs, asset_tickers, equity_assets, tlt_asset,
    )
    if verbose:
        print(f"[build] After merge: {merged.shape}")

    # 7. Dirichlet regime probabilities
    rng = np.random.default_rng(seed)
    merged = _add_regime_probs(
        merged, regime_path,
        concentration=dirichlet_concentration,
        noise_floor=dirichlet_noise_floor,
        rng=rng,
    )

    # 8. Attach Kronos block (if provided)
    if kronos_block_df is not None:
        merged = _attach_kronos_block(merged, kronos_block_df, asset_tickers)
        if verbose:
            print(f"[build] Attached Kronos block: {kronos_block_df.shape}")
    elif verbose:
        print(f"[build] No Kronos block provided; Kronos columns left as NaN")

    # 9. Trim warmup rows
    if trim_warmup > 0:
        merged = merged.iloc[trim_warmup:].reset_index(drop=True)
        if verbose:
            print(f"[build] After trimming {trim_warmup} warmup rows: {merged.shape}")

    return merged


# =============================================================================
# Kronos block construction helper (ticker-prefixes a single aligner output
# applied per asset, OR — more commonly — does one alignment then expands)
# =============================================================================
def build_kronos_block_for_path(synth_regime_path, real_dfs_by_ticker,
                                real_regimes_by_ticker, kronos_columns,
                                asset_tickers=("NVDA", "AMD", "SMH", "TLT"),
                                top_k=5, sample_temperature=0.5, seed=None):
    """
    Build the full ticker-prefixed Kronos block for one synthetic path by
    running KronosAligner once per ticker and concatenating the results
    column-wise with ticker prefixes.

    Why per-ticker? Because each ticker has its own real Kronos history and
    its own real regime sequence — it makes more sense to match each asset
    against its own historical record than to share one aligner.

    Parameters
    ----------
    synth_regime_path : np.ndarray, shape (n_steps,)
    real_dfs_by_ticker : dict[str, pd.DataFrame]
        {ticker: real merged feature DF} — must contain all kronos_columns.
    real_regimes_by_ticker : dict[str, np.ndarray]
        Per-ticker historical regime labels (same encoding as synthetic).
    kronos_columns : list[str]
        Unprefixed Kronos column names.
    asset_tickers : tuple[str]
    top_k, sample_temperature : passed to KronosAligner.assign
    seed : int, optional

    Returns
    -------
    kronos_block : pd.DataFrame, shape (n_steps, 4 * len(kronos_columns))
        Columns are ticker-prefixed: NVDA_kronos_close_d5, AMD_kronos_close_d5, ...
    """
    from kronos_aligner import KronosAligner

    blocks = []
    for i, ticker in enumerate(asset_tickers):
        real_df = real_dfs_by_ticker[ticker]
        real_regimes = real_regimes_by_ticker[ticker]

        aligner = KronosAligner(
            real_features_df=real_df,
            real_regime_seq=real_regimes,
            kronos_columns=kronos_columns,
            seed=None if seed is None else seed + i,  # decorrelate across tickers
        )
        block = aligner.assign(
            synth_regime_path,
            strategy="regime_match",
            top_k=top_k,
            sample_temperature=sample_temperature,
        )
        block = block.rename(columns={c: f"{ticker}_{c}" for c in kronos_columns})
        blocks.append(block)

    return pd.concat(blocks, axis=1)


# =============================================================================
# Schema validation
# =============================================================================
def validate_schema(synth_df, reference_csv_path):
    """
    Verify that synth_df has exactly the same columns (and order) as the real
    merged file. Raises with a diff if not.
    """
    ref_cols = pd.read_csv(reference_csv_path, nrows=0).columns.tolist()
    synth_cols = synth_df.columns.tolist()

    missing = [c for c in ref_cols if c not in synth_cols]
    extra = [c for c in synth_cols if c not in ref_cols]

    if missing or extra:
        msg = []
        if missing:
            msg.append(f"Missing in synthetic ({len(missing)}): {missing[:10]}"
                       + ("..." if len(missing) > 10 else ""))
        if extra:
            msg.append(f"Extra in synthetic ({len(extra)}): {extra[:10]}"
                       + ("..." if len(extra) > 10 else ""))
        raise AssertionError("Schema mismatch:\n  " + "\n  ".join(msg))

    # Also check column order
    if synth_cols != ref_cols:
        # find first divergence
        for i, (a, b) in enumerate(zip(synth_cols, ref_cols)):
            if a != b:
                raise AssertionError(
                    f"Column order mismatch at index {i}: "
                    f"synth has '{a}', reference has '{b}'"
                )

    return True


def reorder_to_reference(synth_df, reference_csv_path):
    """Reorder columns of synth_df to match the reference CSV column order."""
    ref_cols = pd.read_csv(reference_csv_path, nrows=0).columns.tolist()
    missing = [c for c in ref_cols if c not in synth_df.columns]
    if missing:
        raise KeyError(f"Cannot reorder: missing columns {missing[:5]}...")
    return synth_df[ref_cols].copy()
