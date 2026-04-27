"""
Merge per-asset feature DataFrames into a single wide DataFrame for RL training.

Input:  4 per-asset DataFrames (NVDA, AMD, SMH, TLT) each with ~99 columns
        (per-asset features + cross-asset correlation features).

Output: Single date-indexed DataFrame where:
        - Per-asset features are prefixed with ticker (NVDA_RSI_norm_14, AMD_RSI_norm_14, ...)
        - Cross-asset correlation features appear once (no prefix)
        - Single 'date' column as join key
        - close prices kept per asset (NVDA_close, AMD_close, ...) for P&L simulation

Total columns ≈ 4 × (per-asset feature count) + (cross-asset feature count) + 1 (date)

Example
-------
    from merge_asset_features import merge_asset_features
    
    asset_dfs = {
        'NVDA': pd.read_csv('NVDA_RL_Final_Trimmed.csv'),
        'AMD':  pd.read_csv('AMD_RL_Final_Trimmed.csv'),
        'SMH':  pd.read_csv('SMH_RL_Final_Trimmed.csv'),
        'TLT':  pd.read_csv('TLT_RL_Final_Trimmed.csv'),
    }
    merged = merge_asset_features(asset_dfs, save_to='data/RL_Final_Merged.csv')
"""

import pandas as pd
import numpy as np
from pathlib import Path


def merge_asset_features(
    data,
    asset_tickers=('NVDA', 'AMD', 'SMH', 'TLT'),
    cross_asset_ticker='COR',
    date_col='date',
    drop_cols=('TICKER', 'index'),
    save_to=None,
    verbose=True,
):
    """
    Merge per-asset feature DataFrames + a separate cross-asset DataFrame
    into one wide DataFrame for RL training.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        {ticker: df} dictionary. Must contain all assets in `asset_tickers`
        plus the cross-asset DataFrame keyed by `cross_asset_ticker`.
    asset_tickers : tuple[str]
        Per-asset DataFrames to merge with ticker-prefixed columns.
    cross_asset_ticker : str
        Key in the dict for the cross-asset features DataFrame.
        Its columns are merged WITHOUT prefix (they are shared across all assets).
    date_col : str
        Date column name (must be present in every DataFrame).
    drop_cols : tuple[str]
        Columns to drop from each DataFrame before merging.
    save_to : str or Path, optional
        If given, save the merged DataFrame to CSV.
    verbose : bool

    Returns
    -------
    merged : pd.DataFrame
    """
    if verbose:
        print(f"\nMerging {len(asset_tickers)} per-asset DFs + 2 cross-asset DF "
              f"('{cross_asset_ticker}')")

    # Validation: all required keys present
    missing = [t for t in (*asset_tickers, cross_asset_ticker) if t not in data]
    if missing:
        raise KeyError(f"Missing keys in data dict: {missing}")

    # Validation: all DataFrames have same row count and dates
    n_rows = None
    dates_ref = None
    for ticker in (*asset_tickers, cross_asset_ticker):
        df = data[ticker]
        if n_rows is None:
            n_rows = len(df)
            dates_ref = pd.to_datetime(df[date_col]).reset_index(drop=True)
        elif len(df) != n_rows:
            raise ValueError(
                f"Row count mismatch: {ticker} has {len(df)} rows, "
                f"first DF had {n_rows}."
            )
        elif not pd.to_datetime(df[date_col]).reset_index(drop=True).equals(dates_ref):
            raise ValueError(f"Date mismatch in {ticker}.")

    if verbose:
        print(f"  All DFs aligned: {n_rows} rows, "
              f"{dates_ref.min().date()} → {dates_ref.max().date()}")

    # Start the merge with the date column as the canonical reference
    merged = pd.DataFrame({date_col: dates_ref.values})

    # Step 1: Add per-asset blocks (with ticker prefix)
    for ticker in asset_tickers:
        df = data[ticker].copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Drop unwanted columns
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=c)

        # Identify columns to prefix (everything except the date)
        per_asset_cols = [c for c in df.columns if c != date_col]

        # Apply ticker prefix
        rename_map = {c: f"{ticker}_{c}" for c in per_asset_cols}
        df = df.rename(columns=rename_map)
        prefixed_cols = list(rename_map.values())

        # Merge on date
        merged = merged.merge(df[[date_col] + prefixed_cols], on=date_col, how='left')

        if verbose:
            print(f"  [{ticker}] added {len(prefixed_cols)} prefixed columns")

    # Step 2: Add cross-asset block (NO prefix)
    #for ticker in cross_asset_ticker:
    cor_df = data[cross_asset_ticker].copy()
    cor_df[date_col] = pd.to_datetime(cor_df[date_col])

    # Drop unwanted columns from COR too (TICKER, index)
    for c in drop_cols:
        if c in cor_df.columns:
            cor_df = cor_df.drop(columns=c)

    cross_cols = [c for c in cor_df.columns if c != date_col]
    merged = merged.merge(cor_df[[date_col] + cross_cols], on=date_col, how='left')

    if verbose:
        print(f"  [{cross_asset_ticker}] added {len(cross_cols)} cross-asset columns "
              f"(no prefix)")

    # Final reporting
    n_nan = merged.isna().sum().sum()
    if verbose:
        print(f"\nFinal shape: {merged.shape}")
        print(f"Date range:  {merged[date_col].min().date()} → "
              f"{merged[date_col].max().date()}")
        print(f"Total NaN:   {n_nan}")
        if n_nan > 0:
            nan_cols = merged.isna().sum()
            nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
            print(f"NaN columns (top 10):")
            print(nan_cols.head(10).to_string())

    if save_to is not None:
        from pathlib import Path
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(save_to, index=False)
        if verbose:
            print(f"\nSaved to: {save_to}")

    return merged

def column_groups(merged_df, asset_tickers=('NVDA', 'AMD', 'SMH', 'TLT'),
                  cross_asset_prefix='corr_'):
    return {
        'date': [c for c in merged_df.columns if c.lower() == 'date'],
        'per_asset': {
            t: [c for c in merged_df.columns if c.startswith(f"{t}_")]
            for t in asset_tickers
        },
        # Match by prefix since COR's columns aren't prefixed with COR_
        'cross_asset': [c for c in merged_df.columns if c.startswith(cross_asset_prefix)],
    }


if __name__ == "__main__":
    # Demo
    import os
    base = '/mnt/user-data/uploads'

    # If you have your trimmed files locally, change this path
    asset_dfs = {
        ticker: pd.read_csv(os.path.join(base, f'{ticker}_RL_RealData_Features.csv'))
        for ticker in ['NVDA', 'AMD', 'SMH', 'TLT']
    }

    # Trim to common date range first if needed
    common_dates = set.intersection(*[set(df['date']) for df in asset_dfs.values()])
    for ticker in asset_dfs:
        df = asset_dfs[ticker]
        asset_dfs[ticker] = df[df['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)

    merged = merge_asset_features(asset_dfs)
    print(f"\nFirst 3 NVDA-prefixed columns: "
          f"{[c for c in merged.columns if c.startswith('NVDA_')][:3]}")
    print(f"First 3 cross-asset columns: "
          f"{[c for c in merged.columns if c.startswith('corr_')][:3]}")

    groups = column_groups(merged)
    print(f"\nColumn groups summary:")
    for ticker, cols in groups['per_asset'].items():
        print(f"  {ticker}: {len(cols)} per-asset columns")
    print(f"  cross_asset: {len(groups['cross_asset'])} columns")
