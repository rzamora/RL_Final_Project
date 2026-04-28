"""
Production pool builder for RL training.

Wires the per-path synthetic feature builder (synthetic_feature_builder.py) into
a batch pool generator that:
    1. Loads fitted regime/GARCH/copula parameters from pickle
    2. Loads real Kronos features from the merged train CSV (for alignment)
    3. Calls simulate_hybrid_paths once for all paths (efficient — single RNG, single fit)
    4. Per path: builds Kronos block via per-ticker regime matching, then calls
       build_synthetic_features which attaches Kronos and trims warmup
    5. Stacks paths into a tensor, saves as compressed npz
    6. Optional: validates each saved path against RL_Final_Merged_train.csv schema

Why this supersedes build_synthetic_pool.py:
    The original pool builder's `feature_builder` callable contract did not include
    the regime path or Kronos attachment — it expected to do Kronos via a separate
    KronosAligner call and concat after. The new synthetic_feature_builder.py
    does Kronos attachment internally because the merge step needs Kronos columns
    in their canonical positions before column-order validation. So this v2 pool
    builder calls build_synthetic_features directly and lets it own the Kronos step.

Save schema
-----------
features:      (n_paths, n_clean_steps, n_features) float32
feature_names: (n_features,) string array — column name lookup
returns:       (n_paths, n_clean_steps, n_assets) float32 — TRIMMED to match features
prices:        (n_paths, n_clean_steps + 1, n_assets) float32 — TRIMMED to match
volumes:       (n_paths, n_clean_steps, n_assets) float32 — TRIMMED to match
regimes:       (n_paths, n_clean_steps) int8 — TRIMMED to match
metadata:      object array containing dict of build params

`n_clean_steps = n_steps - trim_warmup`. All arrays are aligned in time so that
features[p, t] corresponds to returns[p, t], prices[p, t+1] (post-return close),
volumes[p, t], regimes[p, t] for the same trimmed-time-axis t.
"""

from __future__ import annotations

import pickle
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
)


# =============================================================================
# Real-data loading
# =============================================================================
def load_fitted_pickle(pickle_path):
    """
    Load the fitted GARCH/regime/copula parameters that the simulator needs.

    Expected keys (per regime_dcc_garch_copula_V1.main):
        regime_models   — list of per-regime fitted models
        trans_mat       — Markov transition matrix
        regime_seq      — np.ndarray of historical regime labels per real date
        regime_probs    — historical regime posterior probabilities (unused here)
        training_dates  — pd.DatetimeIndex of real training dates
    """
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Fitted pickle not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        params = pickle.load(f)

    required = {"regime_models", "trans_mat", "regime_seq", "training_dates"}
    missing = required - set(params.keys())
    if missing:
        raise KeyError(f"Pickle missing required keys: {missing}")

    return params


def split_merged_into_per_ticker_kronos(
    merged_df, asset_tickers, kronos_columns,
):
    """
    Take the wide merged train DF (with columns like 'NVDA_kronos_close_d5')
    and produce {ticker: df} where each df has unprefixed Kronos columns,
    in the order specified by `kronos_columns`.

    The KronosAligner inside build_kronos_block_for_path expects unprefixed
    Kronos column names (it adds the ticker prefix itself when assembling the
    final block).

    Parameters
    ----------
    merged_df : pd.DataFrame
        Real merged training DataFrame. Must contain a 'date' column and
        ticker-prefixed Kronos columns for every ticker × Kronos col combo.
    asset_tickers : tuple[str]
    kronos_columns : list[str]
        Unprefixed Kronos column names (the canonical KRONOS_COLUMNS list).

    Returns
    -------
    real_dfs_by_ticker : dict[str, pd.DataFrame]
        Each df has 'date' plus all unprefixed Kronos columns, same row count
        as merged_df.
    """
    if "date" not in merged_df.columns:
        raise KeyError("merged_df must have a 'date' column")

    real_dfs_by_ticker = {}
    for ticker in asset_tickers:
        cols_needed = [f"{ticker}_{c}" for c in kronos_columns]
        missing = [c for c in cols_needed if c not in merged_df.columns]
        if missing:
            raise KeyError(
                f"merged_df missing prefixed Kronos columns for {ticker}: "
                f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
            )

        sub = merged_df[["date"] + cols_needed].copy()
        # Strip the ticker prefix
        rename_map = {f"{ticker}_{c}": c for c in kronos_columns}
        sub = sub.rename(columns=rename_map)
        real_dfs_by_ticker[ticker] = sub.reset_index(drop=True)

    return real_dfs_by_ticker


def build_per_ticker_regime_dict(regime_seq, asset_tickers):
    """
    Build {ticker: regime_seq} dict for build_kronos_block_for_path.

    All tickers share the same regime sequence because regimes are fit on the
    joint return panel (one regime per date, applies to all assets). This
    function exists to make the data shape match what build_kronos_block_for_path
    expects without ambiguity at the call site.
    """
    return {ticker: np.asarray(regime_seq, dtype=int) for ticker in asset_tickers}


def _align_regimes_to_dates(regime_seq_full, training_dates_full, target_dates,
                             verbose=False):
    """
    Pull the regime label for each `target_date` from the full historical
    regime sequence by date-based lookup.

    The fitted pickle stores regime_seq aligned to training_dates_full (the
    entire real history download_data returned). The merged train CSV is
    typically a subset (train/test split), so we can't assume positional
    alignment. We do an exact-date join.

    Parameters
    ----------
    regime_seq_full : np.ndarray, shape (N,) of int
        Regime labels per real date.
    training_dates_full : pd.DatetimeIndex, length N
        Dates corresponding to regime_seq_full[i].
    target_dates : pd.DatetimeIndex
        The dates we want regime labels for (e.g., from the merged CSV).
    verbose : bool

    Returns
    -------
    aligned : np.ndarray, shape (len(target_dates),) of int
        regime label for each target date.

    Raises
    ------
    KeyError
        If any target date is not in training_dates_full. This means the
        merged CSV contains dates the regime model wasn't fit on — usually a
        sign of a stale pickle or an off-by-one in the train/test split.
    """
    # Map full date → regime label. Both indexes are normalized to midnight
    # so we don't get false misses from time-of-day differences.
    full_idx = pd.DatetimeIndex(training_dates_full).normalize()
    target_idx = pd.DatetimeIndex(target_dates).normalize()

    date_to_regime = pd.Series(regime_seq_full, index=full_idx)

    # Detect missing dates up-front for a clean error message
    missing = target_idx.difference(full_idx)
    if len(missing) > 0:
        sample = [d.date().isoformat() for d in missing[:5]]
        raise KeyError(
            f"{len(missing)} date(s) in the merged CSV are not in the fitted "
            f"pickle's training_dates. The regime model wasn't fit on these "
            f"dates — likely a stale pickle or a train/test split mismatch.\n"
            f"  First missing: {sample}\n"
            f"  CSV range:     {target_idx.min().date()} → {target_idx.max().date()}\n"
            f"  Pickle range:  {full_idx.min().date()} → {full_idx.max().date()}"
        )

    aligned = date_to_regime.reindex(target_idx).values.astype(int)

    if verbose and len(target_idx) < len(full_idx):
        n_dropped = len(full_idx) - len(target_idx)
        print(f"    Aligned regimes: kept {len(target_idx)} of "
              f"{len(full_idx)} fitted dates (dropped {n_dropped} not in CSV)")

    return aligned


# Regime label order used by the regime simulator and the Dirichlet probability
# columns. Kept here for the diagnostic display only — not used for anything
# functional.
_REGIME_NAMES = ["Bull", "Bear", "SevereBear", "Crisis"]


def _print_regime_distribution_diagnostic(full_seq, sliced_seq, n_regimes):
    """
    Print a side-by-side comparison of regime distribution in the full pickle
    vs the CSV slice. Flags any regime whose share dropped by >2 percentage
    points after slicing — those are the regimes the Kronos aligner will have
    fewer real windows to draw from.
    """
    full_seq = np.asarray(full_seq, dtype=int)
    sliced_seq = np.asarray(sliced_seq, dtype=int)

    print(f"\n    Regime distribution: pickle full vs CSV slice")
    print(f"    {'regime':<22s} {'pickle':>10s}  {'CSV':>10s}  {'delta':>8s}  flag")

    flagged = []
    for r in range(n_regimes):
        name = _REGIME_NAMES[r] if r < len(_REGIME_NAMES) else f"regime_{r}"
        p_full = (full_seq == r).mean() if len(full_seq) else 0.0
        p_slice = (sliced_seq == r).mean() if len(sliced_seq) else 0.0
        delta = p_slice - p_full

        # Flag thresholds:
        #   - Any regime that LOST > 2pp of representation in the slice
        #   - Any regime under 2% in the slice (very thin alignment pool)
        flag = ""
        if delta < -0.02:
            flag = "shrunk"
            flagged.append(name)
        elif p_slice < 0.02 and p_full > 0.0:
            flag = "thin"
            flagged.append(name)

        print(f"    {name:<22s} {p_full:>10.1%}  {p_slice:>10.1%}  "
              f"{delta:>+8.1%}  {flag}")

    if flagged:
        print(f"\n    NOTE: {flagged} are under-represented in the CSV slice. "
              f"Synthetic paths with these regimes will get Kronos features "
              f"from a thinner pool of real windows.")
    else:
        print(f"    All regimes well-represented in the CSV slice.")


def _print_kronos_match_quality_sample(all_regimes, real_dfs_by_ticker,
                                        real_regimes_by_ticker, asset_tickers,
                                        top_k, sample_temperature,
                                        n_sample_paths, seed):
    """
    Run the Kronos aligner on a handful of paths just to read out match_pct.
    This is cheap (the aligner does a vectorized sliding-window comparison) —
    typically < 1ms per path — and lets us flag alignment problems before
    the expensive per-path feature build runs.

    A healthy match_pct depends on regime persistence and the size of the real
    window pool. For a synthetic path of 512 timesteps against a CSV with
    ~4500 candidate start positions, expect mean match_pct in [0.65, 0.85]
    in normal cases. < 0.55 means the CSV is too narrow — synthetic paths
    contain regime mixtures that don't appear in any real window.
    """
    from kronos_aligner import KronosAligner

    print(f"\n    Kronos match quality (sample of {n_sample_paths} paths, "
          f"per ticker):")
    print(f"    {'ticker':<8s} {'mean':>8s}  {'min':>8s}  {'max':>8s}")

    for ticker in asset_tickers:
        # The per-ticker DFs from split_merged_into_per_ticker_kronos have:
        #   ['date'] + KRONOS_COLUMNS (unprefixed)
        # so we pass KRONOS_COLUMNS directly rather than recomputing from
        # df.columns (which would risk including 'date' or column-order drift).
        aligner = KronosAligner(
            real_features_df=real_dfs_by_ticker[ticker],
            real_regime_seq=real_regimes_by_ticker[ticker],
            kronos_columns=KRONOS_COLUMNS,
            seed=seed,
        )

        match_pcts = []
        for p in range(n_sample_paths):
            block = aligner.assign(
                all_regimes[p],
                strategy="regime_match",
                top_k=top_k,
                sample_temperature=sample_temperature,
            )
            mp = block.attrs.get("match_pct")
            if mp is not None:
                match_pcts.append(mp)

        if match_pcts:
            mean = float(np.mean(match_pcts))
            mn = float(np.min(match_pcts))
            mx = float(np.max(match_pcts))
            flag = "  LOW" if mean < 0.55 else ""
            print(f"    {ticker:<8s} {mean:>8.1%}  {mn:>8.1%}  {mx:>8.1%}{flag}")
        else:
            print(f"    {ticker:<8s} (no match_pct attr — aligner version mismatch?)")


# =============================================================================
# Main pool builder
# =============================================================================
def build_pool(
    *,
    fitted_pickle_path,
    merged_train_csv_path,
    n_paths,
    n_steps,
    seed,
    asset_tickers=("NVDA", "AMD", "SMH", "TLT"),
    equity_assets=("NVDA", "AMD", "SMH"),
    tlt_asset="TLT",
    trim_warmup=128,
    dirichlet_concentration=50.0,
    dirichlet_noise_floor=0.5,
    kronos_top_k=5,
    kronos_sample_temperature=0.5,
    stress_bias=None,
    initial_regime="random",
    start_date="2020-01-01",
    save_to=None,
    validate_against_reference=True,
    reorder_columns_to_reference=True,
    checkpoint_every=None,
    checkpoint_dir=None,
    verbose=True,
):
    """
    Build a synthetic training pool with full features.

    Parameters
    ----------
    fitted_pickle_path : str or Path
        Path to synthetic_generator_FITTED.pkl produced by
        regime_dcc_garch_copula_V1.main (contains regime_models, trans_mat, regime_seq).
    merged_train_csv_path : str or Path
        Path to RL_Final_Merged_train.csv. Used as the source of real Kronos
        features (sliced per-ticker for regime-matched alignment) and as the
        schema reference for validation.
    n_paths : int
        Number of synthetic paths to generate.
    n_steps : int
        Length of each raw simulated path (BEFORE warmup trim). Should be
        >= trim_warmup + desired_clean_length.
    seed : int
        Master RNG seed. Each path gets seed + path_idx for Dirichlet/Kronos
        determinism.
    asset_tickers, equity_assets, tlt_asset
        Same conventions as build_synthetic_features.
    trim_warmup : int
        Rows trimmed from the start of each path. Must be >= 128 to clear the
        wavelet warmup; recommended 128 unless you want to discard more for
        extra safety.
    dirichlet_concentration, dirichlet_noise_floor : float
        Passed to the regime probability column generator.
    kronos_top_k, kronos_sample_temperature : int, float
        Passed to KronosAligner.assign for regime-matched window sampling.
    stress_bias : dict or None
        Optional regime transition bias (e.g., {2: 1.5, 3: 1.5} to oversample
        stressed regimes). Passed straight to simulate_hybrid_paths.
    initial_regime : "random" or int
        Starting regime for each simulated path.
    start_date : str
        Fake business-day index start (cosmetic, doesn't affect features).
    save_to : str, Path, or None
        Where to save the .npz. If None, returns pool dict without saving.
    validate_against_reference : bool
        If True, validates each path's columns match the merged train CSV exactly.
    reorder_columns_to_reference : bool
        If True, reorders each path's columns to match the reference CSV before
        stacking. Highly recommended to ensure feature_names ordering matches
        what the RL agent will expect at inference.
    checkpoint_every : int or None
        If set, saves a partial .npz every N paths (lets you recover from crashes
        on long runs). Requires checkpoint_dir.
    checkpoint_dir : str, Path, or None
        Directory for partial checkpoints.
    verbose : bool

    Returns
    -------
    pool : dict
        See module docstring for schema.
    """
    if checkpoint_every is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_every requires checkpoint_dir")

    # -------------------------------------------------------------------------
    # 1. Load fitted parameters and real data
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print(f"build_pool: n_paths={n_paths}, n_steps={n_steps}, "
              f"trim_warmup={trim_warmup}, seed={seed}")
        print(f"  clean rows per path: {n_steps - trim_warmup}")
        print("=" * 70)
        print(f"\n[1] Loading fitted parameters from {fitted_pickle_path}")

    params = load_fitted_pickle(fitted_pickle_path)
    regime_models = params["regime_models"]
    trans_mat = params["trans_mat"]
    real_regime_seq_full = np.asarray(params["regime_seq"], dtype=int)
    training_dates_full = pd.DatetimeIndex(params["training_dates"]).normalize()

    if len(real_regime_seq_full) != len(training_dates_full):
        raise ValueError(
            f"Pickle inconsistency: regime_seq has {len(real_regime_seq_full)} "
            f"entries but training_dates has {len(training_dates_full)}. The "
            f"pickle is malformed."
        )

    if verbose:
        print(f"    regime_models: {len(regime_models)} regimes")
        print(f"    trans_mat: shape {np.asarray(trans_mat).shape}")
        print(f"    real_regime_seq: {len(real_regime_seq_full)} days "
              f"({training_dates_full.min().date()} → "
              f"{training_dates_full.max().date()})")

    if verbose:
        print(f"\n[2] Loading real Kronos source from {merged_train_csv_path}")
    merged_real = pd.read_csv(merged_train_csv_path, parse_dates=["date"])
    merged_real["date"] = pd.to_datetime(merged_real["date"]).dt.normalize()

    # Align regime_seq to the merged-CSV dates by date join.
    # The CSV is typically the train slice (shorter than the full real history
    # the regimes were fit on); we subset regime_seq to the dates that appear
    # in the CSV. This is also robust to the CSV being filtered/reordered.
    real_regime_seq = _align_regimes_to_dates(
        regime_seq_full=real_regime_seq_full,
        training_dates_full=training_dates_full,
        target_dates=pd.DatetimeIndex(merged_real["date"]),
        verbose=verbose,
    )

    if verbose:
        print(f"    merged_real: {merged_real.shape}, "
              f"date range {merged_real['date'].min().date()} → "
              f"{merged_real['date'].max().date()}")
        print(f"    Aligned regime sequence: {len(real_regime_seq)} entries "
              f"(matches CSV row count)")

    # Diagnostic: compare regime composition in pickle (full HMM fit window)
    # vs the CSV slice. Large deltas — especially in stressed regimes — mean
    # the warmup-trim chopped off regime-rich periods, and the Kronos aligner
    # will have a thinner pool of real windows to match synthetic stress paths
    # against. Match quality will degrade silently for those segments unless
    # you widen the CSV.
    if verbose:
        _print_regime_distribution_diagnostic(
            full_seq=real_regime_seq_full,
            sliced_seq=real_regime_seq,
            n_regimes=int(np.asarray(trans_mat).shape[0]),
        )

    real_dfs_by_ticker = split_merged_into_per_ticker_kronos(
        merged_real, asset_tickers, KRONOS_COLUMNS,
    )
    real_regimes_by_ticker = build_per_ticker_regime_dict(
        real_regime_seq, asset_tickers,
    )
    if verbose:
        print(f"    Split into per-ticker Kronos histories: "
              f"{list(real_dfs_by_ticker.keys())}")

    # Reference column list (used for validation + reorder)
    ref_cols = merged_real.columns.tolist()
    if verbose:
        print(f"    Reference schema: {len(ref_cols)} columns")

    # -------------------------------------------------------------------------
    # 3. Simulate all paths in one batch
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[3] Simulating {n_paths} paths via simulate_hybrid_paths...")
    t_sim = time.time()

    from regime_dcc_garch_copula_V1 import simulate_hybrid_paths
    all_returns, all_regimes, all_volumes, all_prices = simulate_hybrid_paths(
        regime_models=regime_models,
        trans_mat=trans_mat,
        assets=list(asset_tickers),
        initial_regime=initial_regime,
        n_steps=n_steps,
        n_paths=n_paths,
        return_volumes=True,
        return_prices=True,
        stress_bias=stress_bias,
        seed=seed,
        start_price=100.0,
    )
    if verbose:
        print(f"    Simulated in {time.time() - t_sim:.1f}s — "
              f"returns {all_returns.shape}, prices {all_prices.shape}")

    # Diagnostic: sample Kronos match quality on first few paths.
    # Cheap (~1ms per path), surfaces alignment problems before we burn
    # minutes building features. If average match drops below ~60%, the
    # CSV slice is too short or too narrow for the synthetic regime mixture.
    if verbose:
        _print_kronos_match_quality_sample(
            all_regimes=all_regimes,
            real_dfs_by_ticker=real_dfs_by_ticker,
            real_regimes_by_ticker=real_regimes_by_ticker,
            asset_tickers=asset_tickers,
            top_k=kronos_top_k,
            sample_temperature=kronos_sample_temperature,
            n_sample_paths=min(5, n_paths),
            seed=seed,
        )

    # -------------------------------------------------------------------------
    # 4. Per-path feature build
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[4] Building features for {n_paths} paths...")
    t_build = time.time()

    feature_names = None
    features_list = []

    for p in range(n_paths):
        # 4a. Build per-ticker Kronos block for this path's regime sequence
        kronos_block = build_kronos_block_for_path(
            synth_regime_path=all_regimes[p],
            real_dfs_by_ticker=real_dfs_by_ticker,
            real_regimes_by_ticker=real_regimes_by_ticker,
            kronos_columns=KRONOS_COLUMNS,
            asset_tickers=asset_tickers,
            top_k=kronos_top_k,
            sample_temperature=kronos_sample_temperature,
            seed=seed + p,
        )

        # 4b. Build full features (Kronos attached internally, warmup trimmed)
        synth_df = build_synthetic_features(
            returns_path=all_returns[p],
            volumes_path=all_volumes[p],
            prices_path=all_prices[p],
            regime_path=all_regimes[p],
            asset_tickers=asset_tickers,
            equity_assets=equity_assets,
            tlt_asset=tlt_asset,
            kronos_block_df=kronos_block,
            dirichlet_concentration=dirichlet_concentration,
            dirichlet_noise_floor=dirichlet_noise_floor,
            trim_warmup=trim_warmup,
            start_date=start_date,
            seed=seed + p,
            verbose=False,
        )

        # 4c. Reorder + validate
        if reorder_columns_to_reference:
            try:
                synth_df = synth_df[ref_cols].copy()
            except KeyError as e:
                missing = [c for c in ref_cols if c not in synth_df.columns]
                extra = [c for c in synth_df.columns if c not in ref_cols]
                raise KeyError(
                    f"Path {p}: column reorder failed.\n"
                    f"  Missing in synth ({len(missing)}): {missing[:5]}\n"
                    f"  Extra in synth ({len(extra)}): {extra[:5]}\n"
                    f"  Original error: {e}"
                )

        if validate_against_reference:
            # Cheap check: just compare column lists. Full validate_schema would
            # re-read the CSV header on every path; we already have ref_cols.
            if synth_df.columns.tolist() != ref_cols:
                raise AssertionError(
                    f"Path {p}: schema mismatch after reorder (this should not "
                    f"happen unless reorder_columns_to_reference is False)."
                )

        # 4d. Capture feature names from first path (drop 'date' — kept separately
        #     in the metadata, not used as a numeric feature)
        feature_cols = [c for c in synth_df.columns if c != "date"]
        if feature_names is None:
            feature_names = feature_cols
        elif feature_cols != feature_names:
            raise ValueError(
                f"Path {p}: feature column order differs from path 0. "
                f"This indicates a bug in build_synthetic_features."
            )

        features_list.append(synth_df[feature_cols].values.astype(np.float32))

        if verbose and ((p + 1) % max(1, n_paths // 20) == 0 or p + 1 == n_paths):
            elapsed = time.time() - t_build
            rate = (p + 1) / elapsed
            eta = (n_paths - p - 1) / rate if rate > 0 else 0
            print(f"    [{p+1}/{n_paths}] {rate:.2f} paths/sec, ETA {eta:.0f}s")

        if checkpoint_every is not None and (p + 1) % checkpoint_every == 0:
            _save_checkpoint(
                features_list, all_returns, all_volumes, all_prices, all_regimes,
                feature_names, p + 1, trim_warmup, checkpoint_dir, seed,
            )

    if verbose:
        print(f"    Total feature build time: {time.time() - t_build:.1f}s")

    # -------------------------------------------------------------------------
    # 5. Stack and trim companion arrays to match clean-rows length
    # -------------------------------------------------------------------------
    features_array = np.stack(features_list)  # (n_paths, n_clean_steps, n_features)

    # Trim companion arrays the same way build_synthetic_features trimmed features.
    # build_synthetic_features does: merged.iloc[trim_warmup:].reset_index(drop=True)
    # → drops the first `trim_warmup` rows.
    returns_trimmed = all_returns[:, trim_warmup:, :].astype(np.float32)
    volumes_trimmed = all_volumes[:, trim_warmup:, :].astype(np.float32)
    regimes_trimmed = all_regimes[:, trim_warmup:].astype(np.int8)

    # Prices: shape (n_paths, n_steps+1, N_assets). prices[t+1] is the close at
    # time t (after return r_t). Features at trimmed time t correspond to
    # returns[t + trim_warmup] in raw space, so the matching close is
    # prices[t + trim_warmup + 1] in raw space. We keep prices[trim_warmup:] so
    # that prices_trimmed[0] is the "prior close" before the first clean
    # observation — analogous to the (n_steps+1) shape convention.
    prices_trimmed = all_prices[:, trim_warmup:, :].astype(np.float32)

    # Sanity: lengths must align
    n_clean = features_array.shape[1]
    if returns_trimmed.shape[1] != n_clean:
        raise AssertionError(
            f"Trim mismatch: features have {n_clean} clean rows but "
            f"returns_trimmed has {returns_trimmed.shape[1]}."
        )

    if verbose:
        print(f"\n[5] Final tensor shapes:")
        print(f"    features:  {features_array.shape}")
        print(f"    returns:   {returns_trimmed.shape}")
        print(f"    prices:    {prices_trimmed.shape}")
        print(f"    volumes:   {volumes_trimmed.shape}")
        print(f"    regimes:   {regimes_trimmed.shape}")
        print(f"    n_features: {len(feature_names)}")
        size_mb = features_array.nbytes / 1e6
        print(f"    features tensor size (uncompressed): {size_mb:.1f} MB")

    pool = {
        "features": features_array,
        "feature_names": np.array(feature_names),
        "returns": returns_trimmed,
        "prices": prices_trimmed,
        "volumes": volumes_trimmed,
        "regimes": regimes_trimmed,
        "metadata": np.array([{
            "seed": seed,
            "n_paths": n_paths,
            "n_steps_raw": n_steps,
            "trim_warmup": trim_warmup,
            "n_clean_steps": n_clean,
            "asset_tickers": list(asset_tickers),
            "stress_bias": stress_bias,
            "initial_regime": initial_regime,
            "dirichlet_concentration": dirichlet_concentration,
            "dirichlet_noise_floor": dirichlet_noise_floor,
            "kronos_top_k": kronos_top_k,
            "kronos_sample_temperature": kronos_sample_temperature,
            "fitted_pickle_path": str(fitted_pickle_path),
            "merged_train_csv_path": str(merged_train_csv_path),
        }], dtype=object),
    }

    # -------------------------------------------------------------------------
    # 6. Save
    # -------------------------------------------------------------------------
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"\n[6] Saving to {save_to}...")
        t_save = time.time()
        np.savez_compressed(save_to, **pool)
        if verbose:
            file_mb = save_to.stat().st_size / 1e6
            print(f"    Saved in {time.time() - t_save:.1f}s ({file_mb:.1f} MB)")

    if verbose:
        print("\n" + "=" * 70)
        print("Pool build complete")
        print("=" * 70)

    return pool


# =============================================================================
# Checkpoint helper
# =============================================================================
def _save_checkpoint(features_list, returns_full, volumes_full, prices_full,
                     regimes_full, feature_names, n_done, trim_warmup,
                     checkpoint_dir, seed):
    """Save partial pool — useful for long runs where you want crash recovery."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    features_array = np.stack(features_list)
    out_path = checkpoint_dir / f"checkpoint_seed{seed}_paths{n_done:05d}.npz"
    np.savez_compressed(
        out_path,
        features=features_array,
        feature_names=np.array(feature_names),
        returns=returns_full[:n_done, trim_warmup:, :].astype(np.float32),
        prices=prices_full[:n_done, trim_warmup:, :].astype(np.float32),
        volumes=volumes_full[:n_done, trim_warmup:, :].astype(np.float32),
        regimes=regimes_full[:n_done, trim_warmup:].astype(np.int8),
    )
    print(f"    [checkpoint] saved {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")


# =============================================================================
# Loader + sampler
# =============================================================================
def load_pool(path):
    """
    Load a saved pool. Returns a dict matching build_pool's output schema.

    Notes
    -----
    feature_names comes back as np.ndarray of dtype object/str — convert to
    list if you want to use it for column lookup.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")

    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"],
        "feature_names": data["feature_names"],
        "returns": data["returns"],
        "prices": data["prices"],
        "volumes": data["volumes"],
        "regimes": data["regimes"],
        "metadata": data["metadata"][0] if "metadata" in data else {},
    }


class PoolSampler:
    """
    Wrap a pool for batched access during RL training.

    Usage:
        pool = load_pool('data/synthetic_pool_seed42.npz')
        sampler = PoolSampler(pool)
        rng = np.random.default_rng(0)
        for batch_idx in range(n_batches):
            batch = sampler.sample(batch_size=128, rng=rng)
            features = batch['features']    # (128, n_clean_steps, n_features)
            returns  = batch['returns']     # (128, n_clean_steps, n_assets)
    """

    def __init__(self, pool):
        self.pool = pool
        self.n_paths = pool["features"].shape[0]
        self.feature_names = (
            list(pool["feature_names"])
            if isinstance(pool["feature_names"], np.ndarray)
            else pool["feature_names"]
        )

    def sample(self, batch_size, rng=None, replace=False):
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(self.n_paths, size=batch_size, replace=replace)
        return {
            "features": self.pool["features"][idx],
            "returns": self.pool["returns"][idx],
            "prices": self.pool["prices"][idx],
            "volumes": self.pool["volumes"][idx],
            "regimes": self.pool["regimes"][idx],
            "path_idx": idx,
        }


# =============================================================================
# Round-trip verification helper
# =============================================================================
def verify_round_trip(pool, save_path, verbose=True):
    """
    Save pool to disk, reload, verify all arrays match exactly.
    Use this after building a small test pool to confirm the npz format
    handles your pool cleanly before scaling up.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **pool)

    reloaded = load_pool(save_path)

    # Use equal_nan=True so NaN-in-same-position counts as equal (NaNs in the
    # features tensor are legitimate — wavelet/technical indicators can produce
    # them on regime transitions or constant windows). The npz format preserves
    # NaN positions and float bit patterns; if the saved tensor has NaNs, the
    # reloaded one will too at the same positions.
    checks = {
        "features": np.array_equal(pool["features"], reloaded["features"], equal_nan=True),
        "returns": np.array_equal(pool["returns"], reloaded["returns"], equal_nan=True),
        "prices": np.array_equal(pool["prices"], reloaded["prices"], equal_nan=True),
        "volumes": np.array_equal(pool["volumes"], reloaded["volumes"], equal_nan=True),
        "regimes": np.array_equal(pool["regimes"], reloaded["regimes"]),  # ints, no NaN possible
        "feature_names": list(pool["feature_names"]) == list(reloaded["feature_names"]),
    }

    if verbose:
        print(f"\nRound-trip verification on {save_path.name}:")
        for k, ok in checks.items():
            print(f"  {k:<14s}: {'OK' if ok else 'MISMATCH'}")
        print(f"  metadata keys reloaded: {list(reloaded['metadata'].keys())}")

    if not all(checks.values()):
        bad = [k for k, ok in checks.items() if not ok]
        # Real mismatch — give a precise diff so we know if it's float
        # precision (could happen with savez quirks) or actual content drift
        for key in bad:
            if key == "feature_names":
                orig = list(pool["feature_names"])
                rel = list(reloaded["feature_names"])
                diffs = [(i, a, b) for i, (a, b) in enumerate(zip(orig, rel)) if a != b]
                print(f"  feature_names diff: {len(diffs)} positions, first 3: {diffs[:3]}")
                continue
            a = pool[key]
            b = reloaded[key]
            if a.shape != b.shape:
                print(f"  {key}: shape {a.shape} vs {b.shape}")
                continue
            # Find positions where original and reloaded disagree
            # (treating NaN positions as equal so we isolate true differences)
            both_nan = np.isnan(a) & np.isnan(b) if a.dtype.kind == "f" else None
            differ = (a != b)
            if both_nan is not None:
                differ = differ & ~both_nan
            n_differ = int(differ.sum())
            if n_differ == 0:
                print(f"  {key}: all values match where both finite — "
                      f"the only mismatch was NaN handling. "
                      f"(This shouldn't trigger anymore with equal_nan=True; "
                      f"if you see this, please report.)")
            else:
                # Sample a few real differences to show magnitude
                idx = np.argwhere(differ)[:5]
                print(f"  {key}: {n_differ} positions differ. First 5:")
                for ix in idx:
                    ix_t = tuple(ix)
                    print(f"    at {ix_t}: original={a[ix_t]} vs reloaded={b[ix_t]}")
        raise AssertionError(f"Round-trip failed for: {bad}")

    return reloaded


if __name__ == "__main__":
    print("This module is meant to be imported. See run_pool_build.py for the "
          "production entry point.")
