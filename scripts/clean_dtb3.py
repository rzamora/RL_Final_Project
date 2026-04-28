"""
Clean and normalize DTB3_StockData_RL.csv after manual Excel edits.

Excel saves CSVs with locale-specific date formats (e.g., M/D/YY for US).
The two-digit year is ambiguous: '54' could be 1954 or 2054. This script
resolves that by treating any date in the far future as historical.

Idempotent — running on an already-clean ISO-formatted file leaves it
unchanged.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from project_config import PATHS


def main():
    df = pd.read_csv(PATHS.dtb3_csv)
    print(f"Read {len(df)} rows from {PATHS.dtb3_csv}")

    # Try ISO first — already-clean files
    parsed_iso = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")
    if parsed_iso.notna().sum() >= len(df) * 0.99:
        df["DATE"] = parsed_iso
        print("  Detected ISO format — no conversion needed")
    else:
        # Excel-style M/D/YY format
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", format="%m/%d/%y")
        print("  Detected Excel format (M/D/YY) — converting to ISO")

    n_bad = df["DATE"].isna().sum()
    if n_bad:
        print(f"  Dropping {n_bad} unparseable row(s)")
        df = df.dropna(subset=["DATE"])

    # Fix mis-pivoted dates: %y pivots at 1969, so historical pre-1969 dates
    # parse as 20XX. Anything after 2027 is in this category.
    cutoff = pd.Timestamp("2027-01-01")
    future_mask = df["DATE"] >= cutoff
    n_future = future_mask.sum()
    if n_future:
        print(f"  Correcting {n_future} mis-pivoted historical dates")
        df.loc[future_mask, "DATE"] = df.loc[future_mask, "DATE"] - pd.DateOffset(years=100)

    df = df.sort_values("DATE").drop_duplicates(subset=["DATE"]).reset_index(drop=True)
    df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d")

    print(f"Cleaned: {len(df)} rows, {df['DATE'].iloc[0]} to {df['DATE'].iloc[-1]}")
    df.to_csv(PATHS.dtb3_csv, index=False)
    print(f"✓ Saved to {PATHS.dtb3_csv}")


if __name__ == "__main__":
    main()
