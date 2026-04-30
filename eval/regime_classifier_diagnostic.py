"""
eval/regime_classifier_diagnostic.py

Tests whether regime labels predict realized returns on real_test the same
way they do on real_train.

Hypothesis being tested: the regime classifier degrades out-of-sample. If
true, regime-conditioned policies (HRL) and regime-reweighted training will
both fail because the test labels don't carry the same structural meaning
as the train labels.

Diagnostic: for each regime label, compute the mean realized pct-change per
asset on train days vs test days with that label. If train-Bull NVDA returns
are strongly positive but test-Bull NVDA returns are flat/negative, the label
has lost its predictive content out-of-sample.

CORRECTIONS vs original draft:
  * Return columns are {ASSET}_CPct_Chg1 (not {ASSET}_pct_chg).
  * CPct_Chg1 is stored in PERCENT units (e.g. 0.36 means 0.36%, NVDA daily
    max is +24.36 = +24.36%). The script's downstream logic — thresholds
    (1e-4, 0.001) and the "*100" in print statements — assumes DECIMAL
    units (0.0036 = 0.36%). We divide by 100 on load to convert to
    decimal so the thresholds and print formatting stay correct.
  * project_config dependency removed — paths set explicitly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from project_config import PATHS

TRAIN_CSV = Path(PATHS.train_csv)
TEST_CSV = Path(PATHS.test_csv)

# ---- Paths (replaces project_config.PATHS) -----------------------------------

# ---- Schema constants --------------------------------------------------------
REGIME_NAMES = ["Bull", "Bear", "SevereBear", "Crisis"]
REGIME_PROB_COLS = [
    "regime_prob_Bull",
    "regime_prob_Bear",
    "regime_prob_SevereBear",
    "regime_prob_Crisis",
]
# CPct_Chg1 = 1-day close pct change. Stored in PERCENT units; we divide by
# 100 on load to get decimal units (so 0.0036 = 0.36% daily return).
ASSET_RAW_COLS = ["NVDA_CPct_Chg1", "AMD_CPct_Chg1", "SMH_CPct_Chg1", "TLT_CPct_Chg1"]
ASSET_COLS     = ["NVDA_ret",       "AMD_ret",       "SMH_ret",       "TLT_ret"]
EQUITY_ASSETS    = ["NVDA_ret", "AMD_ret", "SMH_ret"]
DEFENSIVE_ASSETS = ["TLT_ret"]


def load_and_label(csv_path):
    df = pd.read_csv(csv_path)

    missing = [c for c in REGIME_PROB_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"CSV {csv_path.name} missing regime prob columns: {missing}")

    missing_ret = [c for c in ASSET_RAW_COLS if c not in df.columns]
    if missing_ret:
        candidates = [c for c in df.columns if "Chg" in c or "chg" in c]
        raise KeyError(
            f"CSV {csv_path.name} missing return columns {missing_ret}.\n"
            f"  Available chg-like columns: {candidates}"
        )

    # Convert percent -> decimal so downstream thresholds + printing work.
    for raw, clean in zip(ASSET_RAW_COLS, ASSET_COLS):
        df[clean] = df[raw] / 100.0

    regime_probs = df[REGIME_PROB_COLS].to_numpy()
    df["regime"] = regime_probs.argmax(axis=1)
    return df


def per_regime_stats(df, label):
    print(f"\n{'='*88}")
    print(f"DATASET: {label} ({len(df)} days)")
    print(f"{'='*88}")

    print(f"\nRegime distribution:")
    print(f"  {'regime':<12s}  {'count':>6s}  {'pct':>6s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}")
    for r, name in enumerate(REGIME_NAMES):
        n = int((df["regime"] == r).sum())
        pct = n / len(df) * 100
        print(f"  {name:<12s}  {n:>6d}  {pct:>5.1f}%")

    print(f"\nMean daily pct-change by regime × asset (in %):")
    header = "  {:<12s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
        "regime", "NVDA", "AMD", "SMH", "TLT", "EQ_avg"
    )
    print(header)
    print("  " + "-" * 12 + "  " + ("-" * 8 + "  ") * 5)

    per_regime_table = {}
    for r, name in enumerate(REGIME_NAMES):
        mask = df["regime"] == r
        if mask.sum() == 0:
            print(f"  {name:<12s}  (no days)")
            per_regime_table[name] = None
            continue

        sub = df.loc[mask, ASSET_COLS]
        means = sub.mean()
        eq_avg = sub[EQUITY_ASSETS].mean(axis=1).mean()
        per_regime_table[name] = {
            "n": int(mask.sum()),
            "NVDA":   float(means["NVDA_ret"]),
            "AMD":    float(means["AMD_ret"]),
            "SMH":    float(means["SMH_ret"]),
            "TLT":    float(means["TLT_ret"]),
            "EQ_avg": float(eq_avg),
        }
        print(f"  {name:<12s}  {means['NVDA_ret']*100:>+7.3f}%  "
              f"{means['AMD_ret']*100:>+7.3f}%  "
              f"{means['SMH_ret']*100:>+7.3f}%  "
              f"{means['TLT_ret']*100:>+7.3f}%  "
              f"{eq_avg*100:>+7.3f}%")

    print(f"\nHit rate (% of days with positive equity-avg pct-change) by regime:")
    print(f"  {'regime':<12s}  {'hit_rate':>10s}")
    print(f"  {'-'*12}  {'-'*10}")
    for r, name in enumerate(REGIME_NAMES):
        mask = df["regime"] == r
        if mask.sum() == 0:
            continue
        eq_avg = df.loc[mask, EQUITY_ASSETS].mean(axis=1)
        hit = (eq_avg > 0).mean()
        print(f"  {name:<12s}  {hit*100:>9.1f}%")

    regime_arr = df["regime"].to_numpy()
    if len(regime_arr) > 1:
        n_changes = int((regime_arr[1:] != regime_arr[:-1]).sum())
        persistence = 1 - n_changes / (len(regime_arr) - 1)
        print(f"\nRegime persistence (P[regime_t == regime_{{t-1}}]): {persistence:.3f}")
        print(f"  Number of regime changes: {n_changes}")

    return per_regime_table


def compare_train_test(train_table, test_table):
    print(f"\n{'='*88}")
    print(f"TRAIN vs TEST COMPARISON: per-regime mean pct-change (in %)")
    print(f"{'='*88}")
    print()
    print(f"  {'regime':<12s}  {'asset':<6s}  {'train_mean':>11s}  "
          f"{'test_mean':>11s}  {'delta':>10s}  flag")
    print(f"  {'-'*12}  {'-'*6}  {'-'*11}  {'-'*11}  {'-'*10}  ----")

    flags = []
    for name in REGIME_NAMES:
        tr = train_table.get(name)
        te = test_table.get(name)
        if tr is None or te is None:
            continue
        for asset in ["NVDA", "AMD", "SMH", "TLT", "EQ_avg"]:
            tr_v = tr[asset]
            te_v = te[asset]
            delta = te_v - tr_v

            sign_flip = (tr_v * te_v < 0) and abs(tr_v) > 1e-4 and abs(te_v) > 1e-4
            collapse = (abs(tr_v) > 0.001) and (abs(te_v) < 0.25 * abs(tr_v))

            flag = ""
            if sign_flip:
                flag = "FLIP"
                flags.append((name, asset, "sign flip", tr_v, te_v))
            elif collapse:
                flag = "weak"
                flags.append((name, asset, "magnitude collapse", tr_v, te_v))

            print(f"  {name:<12s}  {asset:<6s}  {tr_v*100:>+10.3f}%  "
                  f"{te_v*100:>+10.3f}%  {delta*100:>+9.3f}%  {flag}")
        print()

    return flags


def summarize_findings(flags, train_table, test_table):
    print(f"{'='*88}")
    print(f"SUMMARY")
    print(f"{'='*88}")
    print()

    if not flags:
        print("NO sign flips or magnitude collapses detected.")
        print("Regime labels appear to retain predictive content out-of-sample.")
        print("→ regime classifier is NOT the bottleneck.")
        print("→ Phase 3 reweighting underperformance is more likely a data-quantity issue.")
        print("→ Recommend Step B: regenerate synth pool with stress_bias.")
        return

    sign_flips = [f for f in flags if f[2] == "sign flip"]
    collapses  = [f for f in flags if f[2] == "magnitude collapse"]

    print(f"Found {len(sign_flips)} sign flip(s) and {len(collapses)} magnitude collapse(s):")
    print()

    if sign_flips:
        print("  SIGN FLIPS (train vs test mean returns have opposite sign):")
        for name, asset, _, tr_v, te_v in sign_flips:
            print(f"    {name:<12s} × {asset:<6s}  train={tr_v*100:+.3f}%  test={te_v*100:+.3f}%")
        print()

    if collapses:
        print("  MAGNITUDE COLLAPSES (test mean is <25% of train mean):")
        for name, asset, _, tr_v, te_v in collapses:
            print(f"    {name:<12s} × {asset:<6s}  train={tr_v*100:+.3f}%  test={te_v*100:+.3f}%")
        print()

    critical = []
    for name, asset, kind, tr_v, te_v in flags:
        if asset in ["NVDA", "AMD", "SMH", "EQ_avg"] and name in ["Bull", "SevereBear"]:
            if kind == "sign flip":
                critical.append((name, asset, kind))

    if critical:
        print("CRITICAL: sign flips found on equity assets in Bull/SB regimes:")
        for name, asset, kind in critical:
            print(f"    {name} regime predicts WRONG direction for {asset} on test")
        print()
        print("→ The regime classifier IS the bottleneck.")
        print("→ The HL of any HRL is being trained to associate regime labels with")
        print("  market behaviors that don't hold on test.")
        print("→ Reweighting / regenerating synth data will NOT fix this.")
        print("→ Recommend: write up the negative result (Step E from Phase 3 report)")
        print("  OR investigate replacing the regime classifier (Step F).")
    else:
        print("Found weakening of regime signal but no critical sign flips on equity assets.")
        print("→ Regime classifier is degraded but not catastrophic.")
        print("→ Reweighting + more data may still help. Recommend Step B then re-evaluate.")


def main():
    print("=" * 88)
    print("Regime classifier diagnostic")
    print("=" * 88)
    print()
    print("Testing whether regime labels predict realized returns the same way")
    print("on real_test as on real_train.")

    print(f"\nLoading {TRAIN_CSV.name}...")
    train_df = load_and_label(TRAIN_CSV)
    print(f"  {len(train_df)} rows")

    print(f"Loading {TEST_CSV.name}...")
    test_df = load_and_label(TEST_CSV)
    print(f"  {len(test_df)} rows")

    train_table = per_regime_stats(train_df, "real_train (2004-2022)")
    test_table  = per_regime_stats(test_df,  "real_test (2023-2026)")

    flags = compare_train_test(train_table, test_table)

    summarize_findings(flags, train_table, test_table)


if __name__ == "__main__":
    main()
