"""
Cross-asset correlation features.

Computes rolling pairwise correlations and aggregate breadth signals from the
daily returns of multiple assets, then merges these features back into each
asset's feature DataFrame.

Output features (added to every asset's DataFrame):
    Pairwise (6 pairs × 2 windows × 2 metrics = 24 features):
        corr_{ASSET_A}_{ASSET_B}_w{N}        — rolling Pearson correlation
        corr_{ASSET_A}_{ASSET_B}_w{N}_z      — z-score vs 252d mean

    Aggregate breadth (5 features):
        corr_eq_avg_w20                      — avg correlation among equities (NVDA/AMD/SMH)
        corr_eq_avg_w60
        corr_tlt_eq_avg_w20                  — avg correlation TLT vs equities
        corr_tlt_eq_avg_w60
        corr_dispersion_w60                  — std of all 6 pair correlations

Total: 29 cross-asset features added to each asset's DataFrame.

Usage
-----
    from cross_asset_correlations import add_correlation_features

    asset_dfs = {
        'NVDA': nvda_df,
        'AMD':  amd_df,
        'SMH':  smh_df,
        'TLT':  tlt_df,
    }
    asset_dfs = add_correlation_features(asset_dfs, return_col='CPct_Chg1')
"""

import numpy as np
import pandas as pd
from itertools import combinations


def _build_returns_panel(asset_dfs, return_col='CPct_Chg1', date_col='date'):
    """
    Build a wide-format DataFrame of daily returns aligned by date.

    Parameters
    ----------
    asset_dfs : dict[str, pd.DataFrame]
        {ticker: df} where each df has a date column and a return column.
    return_col : str
        Name of the daily return column in each df.
    date_col : str
        Name of the date column.

    Returns
    -------
    panel : pd.DataFrame
        Wide-format, date-indexed, columns = tickers, values = daily returns.
        Only rows where ALL assets have a return.
    """
    series_list = []
    for ticker, df in asset_dfs.items():
        print(f"Building returns panel for {ticker}")
        if return_col not in df.columns:
            raise KeyError(f"{ticker}: missing column '{return_col}'")
        if date_col not in df.columns:
            df.reset_index(inplace=True)
            print(
                f"  Warning: {ticker} is missing the date column '{date_col}'; "
                f"renaming to 'date' and re-indexing"
            )
            if date_col not in df.columns:
                raise KeyError(f"{ticker}: missing column '{date_col}'")

        s = (
            df[[date_col, return_col]]
            .dropna()
            .rename(columns={return_col: ticker})
            .assign(**{date_col: lambda x: pd.to_datetime(x[date_col])})
            .set_index(date_col)[ticker]
            .sort_index()
        )
        # Handle duplicate dates by averaging (rare, but defensive)
        if s.index.duplicated().any():
            s = s.groupby(s.index).mean()
            print(f"  Warning: {ticker} has duplicate dates, averaging")
        series_list.append(s)

    panel = pd.concat(series_list, axis=1, join='inner').sort_index()
    panel.index.name = date_col
    return panel


def _compute_pairwise_correlations(returns_panel, windows=(20, 60), zscore_windows=(30, 120)):
    """
    Compute rolling pairwise correlations and z-scores for all asset pairs.

    For each pair × window, computes:
        - corr_{a}_{b}_w{N}            : raw rolling correlation
        - corr_{a}_{b}_w{N}_z{Z}       : z-score using Z-day baseline (one per zscore_window)

    Parameters
    ----------
    zscore_windows : tuple[int]
        One or more baseline window lengths for z-score computation.
        Each adds 12 features (6 pairs × 2 windows).
    """
    # Allow scalar input for backward compatibility
    if isinstance(zscore_windows, (int, float)):
        zscore_windows = (int(zscore_windows),)

    out = pd.DataFrame(index=returns_panel.index)
    pairs = list(combinations(returns_panel.columns, 2))

    for a, b in pairs:
        for w in windows:
            col_corr = f'corr_{a}_{b}_w{w}'
            roll_corr = returns_panel[a].rolling(w).corr(returns_panel[b])
            out[col_corr] = roll_corr.clip(-1, 1)

            # Compute z-score for each baseline window
            for zw in zscore_windows:
                mean_baseline = roll_corr.rolling(zw).mean()
                std_baseline = roll_corr.rolling(zw).std()
                z = (roll_corr - mean_baseline) / (std_baseline + 1e-9)
                out[f'{col_corr}_z{zw}'] = z.clip(-5, 5)

    return out, pairs

def _compute_breadth_features(returns_panel, pairs, equity_assets=None, tlt_asset='TLT',
                              short_window=20, long_window=60):
    """
    Compute aggregate breadth signals across all asset pairs.

    Parameters
    ----------
    equity_assets : list[str], optional
        Tickers considered "equities" for the avg-equity-corr feature.
        Default: every ticker except `tlt_asset`.
    """
    if equity_assets is None:
        equity_assets = [c for c in returns_panel.columns if c != tlt_asset]

    out = pd.DataFrame(index=returns_panel.index)

    eq_pairs = [(a, b) for a, b in pairs if a in equity_assets and b in equity_assets]
    tlt_pairs = [(a, b) for a, b in pairs
                 if (a == tlt_asset or b == tlt_asset)
                 and (a in equity_assets or b in equity_assets)]

    for w in [short_window, long_window]:
        # Average equity-equity correlation
        if eq_pairs:
            cols_eq = [returns_panel[a].rolling(w).corr(returns_panel[b]) for a, b in eq_pairs]
            out[f'corr_eq_avg_w{w}'] = pd.concat(cols_eq, axis=1).mean(axis=1).clip(-1, 1)

        # Average TLT–equity correlation (typically negative; the hedge signal)
        if tlt_pairs:
            cols_tlt = [returns_panel[a].rolling(w).corr(returns_panel[b]) for a, b in tlt_pairs]
            out[f'corr_tlt_eq_avg_w{w}'] = pd.concat(cols_tlt, axis=1).mean(axis=1).clip(-1, 1)

    # Dispersion: std of all 6 pair correlations at the long window
    all_corrs = [returns_panel[a].rolling(long_window).corr(returns_panel[b]) for a, b in pairs]
    out[f'corr_dispersion_w{long_window}'] = pd.concat(all_corrs, axis=1).std(axis=1).clip(0, 1)

    return out


def add_correlation_features(
    asset_dfs,
    return_col='CPct_Chg1',
    date_col='date',
    windows=(20, 60),
    zscore_window=(30, 120),
    equity_assets=None,
    tlt_asset='TLT',
    verbose=True,
):
    """
    Compute cross-asset correlation features and merge them into each asset's DataFrame.

    Parameters
    ----------
    asset_dfs : dict[str, pd.DataFrame]
        {ticker: df} where each df has columns [date_col, return_col, ...features...].
    return_col : str
        Name of daily return column.
    date_col : str
        Name of date column. Same name in every asset df.
    windows : tuple[int]
        Rolling correlation window lengths (in days).
    zscore_window : int
        Window for the z-score baseline of each pair correlation.
    equity_assets : list[str], optional
        Tickers considered equities for breadth aggregates. Defaults to all assets
        except `tlt_asset`.
    tlt_asset : str
        Ticker treated as the hedge asset. Defaults to 'TLT'.
    verbose : bool

    Returns
    -------
    asset_dfs_enriched : dict[str, pd.DataFrame]
        Same keys as input. Each DataFrame has the full set of cross-asset
        correlation features merged in by date.
        Rows where the rolling correlation is undefined will have NaN in the
        new columns (typical for the first ~252 days of the panel).
    """
    if verbose:
        tickers = list(asset_dfs.keys())
        print(f"\nCross-asset correlation features for {len(tickers)} assets: {tickers}")

    # 1. Build aligned returns panel
    panel = _build_returns_panel(asset_dfs, return_col=return_col, date_col=date_col)
    if verbose:
        print(f"  Aligned panel: {len(panel)} dates, "
              f"{panel.index.min().date()} → {panel.index.max().date()}")

    # 2. Pairwise correlations
    pairwise, pairs = _compute_pairwise_correlations(
        panel, windows=windows, zscore_windows=zscore_window
    )
    if verbose:
        print(f"  Computed {len(pairs)} pairwise correlations × {len(windows)} windows × 2 metrics "
              f"= {pairwise.shape[1]} pairwise features")

    # 3. Aggregate breadth
    breadth = _compute_breadth_features(
        panel, pairs,
        equity_assets=equity_assets,
        tlt_asset=tlt_asset,
        short_window=windows[0],
        long_window=windows[-1],
    )
    if verbose:
        print(f"  Computed {breadth.shape[1]} breadth features")

    # 4. Concatenate all cross-asset features
    cross_features = pd.concat([pairwise, breadth], axis=1)
    cross_features.index.name = date_col

    if verbose:
        print(f"  Total cross-asset features: {cross_features.shape[1]}")
        valid_rows = cross_features.dropna()
        print(f"  Rows with full data (post-warmup): {len(valid_rows)}/{len(cross_features)}")

    # 5. Merge back into each asset's DataFrame on date
    enriched = {}
    cross_reset = cross_features.reset_index()
    cross_reset[date_col] = pd.to_datetime(cross_reset[date_col])

    return cross_reset
    # for ticker, df in asset_dfs.items():
    #     df = df.copy()
    #     df[date_col] = pd.to_datetime(df[date_col])
    #     merged = df.merge(cross_reset, on=date_col, how='left')
    #     enriched[ticker] = merged
    #     if verbose:
    #         new_cols = cross_features.shape[1]
    #         total_cols = merged.shape[1]
    #         print(f"    [{ticker}] merged: {merged.shape[0]} rows × {total_cols} cols (+{new_cols} new)")
    #
    # return enriched


def diagnose_correlation_features(enriched_dfs, ticker_to_inspect='NVDA'):
    """
    Quick diagnostic on the new correlation features.
    Prints summary stats and flags problematic features (all-NaN, near-zero variance).
    """
    df = enriched_dfs[ticker_to_inspect]
    corr_cols = [c for c in df.columns if c.startswith('corr_')]
    print(f"\n[{ticker_to_inspect}] Cross-asset correlation feature diagnostics:")
    print(f"  Total cross-asset cols: {len(corr_cols)}")

    sub = df[corr_cols].dropna()
    print(f"  Rows with all features valid: {len(sub)}/{len(df)}")

    stats = sub.describe().T[['mean', 'std', 'min', 'max']].round(3)
    print("\n  Summary stats:")
    print(stats.to_string())

    # Flag potential issues
    near_zero = stats[stats['std'] < 1e-3]
    if len(near_zero) > 0:
        print(f"\n  WARNING: features with near-zero variance: {near_zero.index.tolist()}")


if __name__ == "__main__":
    # Demo
    import os
    base = '/mnt/user-data/uploads'
    asset_dfs = {
        ticker: pd.read_csv(os.path.join(base, f'{ticker}_RL_RealData_Features.csv'))
        for ticker in ['NVDA', 'AMD', 'SMH', 'TLT']
    }

    enriched = add_correlation_features(
        asset_dfs,
        return_col='CPct_Chg1',
        date_col='date',
        windows=(20, 60),
        zscore_window=252,
        equity_assets=['NVDA', 'AMD', 'SMH'],
        tlt_asset='TLT',
    )

    diagnose_correlation_features(enriched, ticker_to_inspect='NVDA')
    print("\n\nLast 3 rows of NVDA correlation features:")
    nvda = enriched['NVDA']
    last_corr_cols = [c for c in nvda.columns if c.startswith('corr_')][:8]
    print(nvda[['date'] + last_corr_cols].tail(3).to_string())
