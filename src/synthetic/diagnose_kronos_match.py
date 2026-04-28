"""
Diagnose why Kronos match quality came in at 54% on the test pool.

Hypothesis: synthetic regime sequences from simulate_hybrid_paths switch
faster (shorter dwell times) than real regime sequences from the HMM fit.
A length mismatch in dwell time means no real 512-day window can cover the
synthetic switching pattern — the aligner is forced to pick the "least bad"
window, which only matches ~half the time.

Run this standalone after a build attempt to confirm or reject the hypothesis,
then act on the result. No changes to the pool builder needed yet — this is
purely investigative.

Usage:
    python diagnose_kronos_match.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Edit if your paths differ
DATA_ROOT = Path("/Users/rafael/Documents/GitHub/RL_Final_Project")
FITTED_PICKLE = DATA_ROOT / "data" / "synthetic" / "models" / "synthetic_generator_FITTED.pkl"
MERGED_CSV = (DATA_ROOT / "data" / "proccessed" / "combined_w_cross_asset"
              / "train" / "RL_Final_Merged_train.csv")

ASSETS = ("NVDA", "AMD", "SMH", "TLT")
N_PATHS = 20
N_STEPS = 512
SEED = 42

REGIME_NAMES = ["Bull", "Bear", "SevereBear", "Crisis"]


def dwell_time_stats(regime_seq):
    """
    Per-regime mean run length: how many consecutive days the chain stays in
    that regime before switching. This is the right unit for comparing
    "switching speed" between real and synthetic.
    """
    seq = np.asarray(regime_seq)
    if len(seq) < 2:
        return {}

    runs = []  # list of (regime, run_length)
    cur = seq[0]
    cur_len = 1
    for x in seq[1:]:
        if x == cur:
            cur_len += 1
        else:
            runs.append((int(cur), cur_len))
            cur = x
            cur_len = 1
    runs.append((int(cur), cur_len))

    by_regime = {}
    for r in np.unique(seq):
        lengths = [length for reg, length in runs if reg == int(r)]
        by_regime[int(r)] = {
            "n_runs": len(lengths),
            "mean_dwell": float(np.mean(lengths)),
            "median_dwell": float(np.median(lengths)),
            "max_dwell": int(np.max(lengths)),
            "share": float((seq == r).mean()),
        }
    return by_regime


def empirical_transition_matrix(regime_seq, n_regimes=4):
    """Per-step transition probabilities, including diagonal (stay)."""
    seq = np.asarray(regime_seq)
    M = np.zeros((n_regimes, n_regimes))
    for a, b in zip(seq[:-1], seq[1:]):
        M[int(a), int(b)] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums


def print_dwell_comparison(real_stats, synth_stats):
    print(f"  {'regime':<14s} {'real mean':>10s} {'synth mean':>11s} "
          f"{'real med':>10s} {'synth med':>10s}  flag")
    for r in sorted(set(real_stats.keys()) | set(synth_stats.keys())):
        name = REGIME_NAMES[r] if r < len(REGIME_NAMES) else f"r{r}"
        rs = real_stats.get(r, {})
        ss = synth_stats.get(r, {})
        rm = rs.get("mean_dwell", 0.0)
        sm = ss.get("mean_dwell", 0.0)
        rmed = rs.get("median_dwell", 0.0)
        smed = ss.get("median_dwell", 0.0)
        ratio = sm / rm if rm > 0 else 0
        flag = ""
        if 0 < ratio < 0.5:
            flag = "synth too fast"
        elif ratio > 2.0:
            flag = "synth too slow"
        print(f"  {name:<14s} {rm:>10.1f} {sm:>11.1f} "
              f"{rmed:>10.1f} {smed:>10.1f}  {flag}")


def print_trans_diag_comparison(real_M, synth_M):
    """Diagonals of the transition matrix = per-step stay probability."""
    print(f"  {'regime':<14s} {'real stay':>10s} {'synth stay':>11s}  delta")
    for r in range(real_M.shape[0]):
        name = REGIME_NAMES[r] if r < len(REGIME_NAMES) else f"r{r}"
        rs = real_M[r, r]
        ss = synth_M[r, r]
        delta = ss - rs
        print(f"  {name:<14s} {rs:>10.3f} {ss:>11.3f}  {delta:+.3f}")


def main():
    print("=" * 70)
    print("Kronos match quality diagnostic")
    print("=" * 70)

    # Load fitted pickle
    with open(FITTED_PICKLE, "rb") as f:
        params = pickle.load(f)

    real_regime_seq_full = np.asarray(params["regime_seq"], dtype=int)
    training_dates_full = pd.DatetimeIndex(params["training_dates"]).normalize()
    trans_mat = np.asarray(params["trans_mat"])

    # Align to CSV slice (same logic as build_pool)
    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    csv_dates = pd.DatetimeIndex(merged["date"])
    date_to_regime = pd.Series(real_regime_seq_full, index=training_dates_full)
    real_regime_csv = date_to_regime.reindex(csv_dates).values.astype(int)

    print(f"\n[1] Real regime sequence (CSV-aligned, {len(real_regime_csv)} days):")
    real_stats = dwell_time_stats(real_regime_csv)
    real_M = empirical_transition_matrix(real_regime_csv)
    for r, s in sorted(real_stats.items()):
        print(f"    {REGIME_NAMES[r]:<14s} share={s['share']:.1%}, "
              f"n_runs={s['n_runs']:>4d}, mean_dwell={s['mean_dwell']:>5.1f} days, "
              f"median={s['median_dwell']:>5.1f}, max={s['max_dwell']}")

    # Generate synthetic regime sequences
    print(f"\n[2] Generating {N_PATHS} synthetic paths × {N_STEPS} steps...")
    import sys
    sys.path.insert(0, str(DATA_ROOT / "src" / "synthetic"))
    from regime_dcc_garch_copula_V1 import simulate_hybrid_paths

    _, all_regimes, _, _ = simulate_hybrid_paths(
        regime_models=params["regime_models"],
        trans_mat=params["trans_mat"],
        assets=list(ASSETS),
        initial_regime="random",
        n_steps=N_STEPS,
        n_paths=N_PATHS,
        return_volumes=True,
        return_prices=True,
        seed=SEED,
        start_price=100.0,
    )
    synth_flat = all_regimes.flatten()
    synth_stats = dwell_time_stats(synth_flat)
    synth_M = empirical_transition_matrix(synth_flat)

    print(f"\n[3] Synthetic regime aggregate ({N_PATHS * N_STEPS} timesteps):")
    for r, s in sorted(synth_stats.items()):
        print(f"    {REGIME_NAMES[r]:<14s} share={s['share']:.1%}, "
              f"n_runs={s['n_runs']:>4d}, mean_dwell={s['mean_dwell']:>5.1f} days, "
              f"median={s['median_dwell']:>5.1f}, max={s['max_dwell']}")

    # The key comparison
    print(f"\n[4] DWELL TIME COMPARISON (key diagnostic):")
    print_dwell_comparison(real_stats, synth_stats)
    print(f"\n    Interpretation:")
    print(f"      'synth too fast' = synthetic regimes switch more often than real.")
    print(f"      The aligner can't find real windows with that switching speed,")
    print(f"      which caps achievable match quality below 100%.")

    print(f"\n[5] PER-STEP STAY PROBABILITY (diagonal of trans matrix):")
    print(f"    Real (empirical from CSV):")
    print_trans_diag_comparison(real_M, synth_M)

    print(f"\n    Trans matrix from pickle (used by simulator):")
    print(f"      diag: {[f'{trans_mat[r,r]:.3f}' for r in range(trans_mat.shape[0])]}")

    # Compute random-baseline match
    p = np.array([s["share"] for r, s in sorted(real_stats.items())])
    random_match = float((p ** 2).sum())
    print(f"\n[6] BASELINES:")
    print(f"    Random alignment match:  {random_match:.1%}")
    print(f"    (Σ p_r² over CSV regime distribution)")
    print(f"    Observed match in build:  ~54%")
    print(f"    Lift over random:         +{(0.54 - random_match)*100:.0f} pp")

    # Distribution-overlap upper bound: even with perfect window selection,
    # the match rate is capped by Σ min(p_real_r, p_synth_r). Intuition: if
    # synth has 5% Crisis but no real window has more than 3% Crisis, the
    # extra 2% of synth Crisis days CANNOT match anywhere — they get aligned
    # to real days that are in some other regime.
    p_real = np.array([s["share"] for r, s in sorted(real_stats.items())])
    p_synth = np.array([s["share"] for r, s in sorted(synth_stats.items())])
    if len(p_real) == len(p_synth):
        overlap_bound = float(np.minimum(p_real, p_synth).sum())
        print(f"    Distribution-overlap upper bound: {overlap_bound:.1%}")
        print(f"    (Σ min(p_real_r, p_synth_r) — match cannot exceed this even")
        print(f"     with perfect window selection, because regimes that are")
        print(f"     more common in synth than real have nowhere to align.)")

        gap_to_bound = overlap_bound - 0.54
        if gap_to_bound > 0.10:
            print(f"\n    Observed match (~54%) is {gap_to_bound:.0%} below the")
            print(f"    distribution-overlap bound. Possible causes:")
            print(f"      (a) Dwell-time mismatch — synth switches at different rate")
            print(f"      (b) Rare-regime entry rate — synth enters stress regimes")
            print(f"          more often than real history contains them, so no real")
            print(f"          window has the right stress-event positions to align to")
            print(f"      (c) Regime co-occurrence patterns differ")
            print(f"    See section [7] below for per-regime breakdown.")
        else:
            print(f"\n    Observed match (~54%) is close to the distribution-")
            print(f"    overlap bound ({overlap_bound:.0%}) — most of the gap to 100%")
            print(f"    is due to regime-share mismatch, not switching speed.")

    # ------------------------------------------------------------------------
    # [7] Per-regime conditional match — the key follow-up diagnostic
    # ------------------------------------------------------------------------
    print(f"\n[7] PER-REGIME CONDITIONAL MATCH:")
    print(f"    For each synthetic regime, what fraction of timesteps in that")
    print(f"    regime have a matching real regime in the chosen alignment window?")
    print(f"    High Bull/Bear + low stress = the rare-regime entry rate problem.")
    print(f"    Uniformly low across all regimes = something else.\n")

    # Re-run the aligner inline so we can harvest the chosen window per path
    # and compute per-regime conditional match.
    sys.path.insert(0, str(DATA_ROOT / "src" / "synthetic"))
    from kronos_aligner import KronosAligner
    from synthetic_feature_builder import KRONOS_COLUMNS

    # Build per-ticker Kronos history and per-ticker regime sequence (same as
    # build_pool does internally).
    merged_real = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    merged_real["date"] = pd.to_datetime(merged_real["date"]).dt.normalize()
    csv_dates = pd.DatetimeIndex(merged_real["date"])
    date_to_regime = pd.Series(real_regime_seq_full, index=training_dates_full)
    real_regime_csv_aligned = date_to_regime.reindex(csv_dates).values.astype(int)

    # Use NVDA — all four tickers will produce identical match scores (same
    # regime sequence, same seed → same chosen window). One ticker is enough.
    ticker = ASSETS[0]
    cols_needed = [f"{ticker}_{c}" for c in KRONOS_COLUMNS]
    sub = merged_real[["date"] + cols_needed].rename(
        columns={f"{ticker}_{c}": c for c in KRONOS_COLUMNS}
    )

    aligner = KronosAligner(
        real_features_df=sub,
        real_regime_seq=real_regime_csv_aligned,
        kronos_columns=KRONOS_COLUMNS,
        seed=SEED,
    )

    # Tally per-regime matches across the sample paths
    per_regime_counts = np.zeros(4, dtype=int)   # synth=r, real==r
    per_regime_totals = np.zeros(4, dtype=int)   # synth==r
    confusion = np.zeros((4, 4), dtype=int)      # synth, real

    n_diag_paths = min(N_PATHS, all_regimes.shape[0])
    for p in range(n_diag_paths):
        synth_seq = all_regimes[p]
        block = aligner.assign(
            synth_seq, strategy="regime_match",
            top_k=5, sample_temperature=0.5,
        )
        start = block.attrs["source_start_idx"]
        real_seq = real_regime_csv_aligned[start : start + len(synth_seq)]

        for r in range(4):
            mask = synth_seq == r
            per_regime_totals[r] += int(mask.sum())
            per_regime_counts[r] += int((real_seq[mask] == r).sum())
            for r2 in range(4):
                confusion[r, r2] += int((real_seq[mask] == r2).sum())

    print(f"    Sampled {n_diag_paths} paths against ticker '{ticker}' history.\n")
    print(f"    {'synth regime':<14s} {'n steps':>10s} {'matched':>10s} "
          f"{'match %':>10s}")
    for r in range(4):
        name = REGIME_NAMES[r] if r < len(REGIME_NAMES) else f"r{r}"
        total = per_regime_totals[r]
        matched = per_regime_counts[r]
        pct = matched / total if total > 0 else 0.0
        print(f"    {name:<14s} {total:>10d} {matched:>10d} {pct:>10.1%}")

    print(f"\n    Confusion matrix (rows = synth regime, cols = real regime aligned to):")
    header = "    " + " " * 14 + "  ".join(f"{REGIME_NAMES[c]:>10s}" for c in range(4))
    print(header)
    for r in range(4):
        name = REGIME_NAMES[r] if r < len(REGIME_NAMES) else f"r{r}"
        row_total = confusion[r, :].sum()
        cells = "  ".join(
            f"{confusion[r, c] / row_total:>10.1%}" if row_total > 0 else f"{'-':>10s}"
            for c in range(4)
        )
        print(f"    {name:<14s}{cells}")

    # Final interpretation
    bull_match = per_regime_counts[0] / max(per_regime_totals[0], 1)
    bear_match = per_regime_counts[1] / max(per_regime_totals[1], 1)
    sb_match = per_regime_counts[2] / max(per_regime_totals[2], 1)
    crisis_match = per_regime_counts[3] / max(per_regime_totals[3], 1)
    normal_avg = (bull_match + bear_match) / 2
    stress_avg = (sb_match + crisis_match) / 2

    print(f"\n    Normal regimes (Bull+Bear) avg match: {normal_avg:.1%}")
    print(f"    Stress regimes (SevereBear+Crisis) avg match: {stress_avg:.1%}")
    if normal_avg > 0.6 and stress_avg < 0.4:
        print(f"\n    DIAGNOSIS: classic rare-regime entry rate issue.")
        print(f"    Kronos features are reliable in normal regimes (~{normal_avg:.0%}),")
        print(f"    poor in stress regimes (~{stress_avg:.0%}).")
        print(f"    Action: this is acceptable for RL training as long as the agent's")
        print(f"    stress-regime decisions don't depend heavily on Kronos signal.")
        print(f"    If they do, consider stratifying Kronos features by regime")
        print(f"    or down-weighting Kronos columns in stress-regime states.")
    elif stress_avg > 0.6:
        print(f"\n    Stress alignment is healthy — match issue is elsewhere.")
    else:
        print(f"\n    Match quality issues are spread across regimes — investigate")
        print(f"    regime co-occurrence patterns or expand the real CSV slice.")

    print("\n" + "=" * 70)
    print("Done. If 'synth too fast' flags appear, see RECOMMENDATIONS below.")
    print("=" * 70)


if __name__ == "__main__":
    main()
