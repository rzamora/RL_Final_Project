# Synthetic Training Pool — Build Report

**Pool file:** `data/synthetic/pools/synthetic_pool_production_n2000_seed43.npz`
**Build date:** 2026-04-28
**Status:** ✅ Production-ready

---

## Build configuration

| Parameter | Value | Notes |
|---|---|---|
| Preset | `production` | |
| Paths | 2,000 | Independent synthetic trajectories |
| Raw steps per path | 584 | Before warmup trim |
| Warmup trimmed | 200 | Clears deepest chained-rolling warmup (corr_w60_z120 = 179 rows) with 21 rows of margin |
| **Clean rows per path** | **384** | ≈ 18 months of trading days |
| Seed (master RNG) | 43 | Each path uses `seed + path_idx` for Dirichlet/Kronos determinism |
| Asset universe | NVDA, AMD, SMH, TLT | Equities = NVDA/AMD/SMH; hedge = TLT |
| Initial regime | random | Drawn from stationary distribution per path |
| Stress bias | None | No oversampling of stressed regimes |
| Dirichlet concentration | 50.0 | Regime probability smoothing |
| Dirichlet noise floor | 0.5 | Minimum mass on non-active regimes |
| Kronos top-k | 5 | Window candidates per alignment |
| Kronos sample temperature | 0.5 | Slight greediness toward best-matching window |

---

## Tensor shapes

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `features` | (2000, 384, 317) | float32 | Full feature tensor (per-asset technicals, wavelets, cross-asset corr, regime probs, Kronos) |
| `feature_names` | (317,) | str | Column names — order matches reference CSV minus the `date` column |
| `returns` | (2000, 384, 4) | float32 | Per-asset daily returns (in %) — trimmed to match features |
| `prices` | (2000, 385, 4) | float32 | Per-asset closes; `prices[t+1]` is close after return `r_t` |
| `volumes` | (2000, 384, 4) | float32 | Per-asset daily volumes |
| `regimes` | (2000, 384) | int8 | Per-step regime label (0=Bull, 1=Bear, 2=SevereBear, 3=Crisis) |
| `metadata` | object | dict | Build provenance: seed, paths, source pickle, source CSV, all config |

**Total feature cells:** 243,456,000
**Uncompressed tensor size:** ~970 MB (features)
**Compressed file size on disk:** ~300–400 MB

---

## Validity checks

### Data integrity

| Check | Result | Notes |
|---|---|---|
| **NaN cells in features** | **0 / 243,456,000 (0.000%)** | trim_warmup=200 cleared all chained-rolling warmups |
| **Inf cells in features** | **0** | No numerical overflow anywhere |
| Regime probability row sums | min=1.0000, max=1.0000, mean=1.0000 | Sums to unity per row, as required |
| Max class prob per row | mean=0.971, min=0.734 | High concentration intentional (Dirichlet smoothing toward dominant regime) |
| Cross-asset correlation range | [-0.964, 0.989] | Within valid [-1, 1], 17 corr columns |
| Unique regimes seen | [0, 1, 2, 3] | All 4 regimes represented |
| Schema match against reference CSV | ✅ 318 columns (= 317 features + `date`) | Column names and order match `RL_Final_Merged_train.csv` |

### Path diversity (RNG sanity)

| Metric | Value | Pass criterion |
|---|---|---|
| `NVDA_close` mean per path — min | $5.47 | Should vary widely |
| `NVDA_close` mean per path — max | $1,319.00 | |
| `NVDA_close` mean per path — std across paths | $106.20 | Must NOT be ~0 |

Wide spread confirms paths are genuinely different — RNG diversity working.

---

## Regime distribution

How the 2,000 × 384 = 768,000 timesteps in the pool break down:

| Regime | Pool share | Real (CSV) | Delta |
|---|---|---|---|
| 0 — Bull | 39.4% | 40.3% | −0.9% |
| 1 — Bear | 36.1% | 36.6% | −0.5% |
| 2 — SevereBear | 17.1% | 16.5% | +0.6% |
| 3 — Crisis | **7.5%** | 6.6% | +0.9% |

Pool distribution within ±1 percentage point of historical reality across all four regimes. Crisis exposure (7.5%) slightly elevated vs real, which is desirable for RL training — gives the agent more stress-regime episodes to learn from than 18 years of real history alone provided.

---

## Kronos alignment quality

Diagnostic from `diagnose_kronos_match.py` on a 20-path sample:

| Synth regime | Match % | Interpretation |
|---|---|---|
| Bull | 60.0% | Reliable — good real-window pool |
| Bear | 65.6% | Reliable — good real-window pool |
| SevereBear | 25.7% | Mostly aligns to real Bear (40.5% of cells) — closest semantic neighbor |
| Crisis | **60.6%** | Reliable despite rarity — 7 distinct real Crisis episodes provide enough match candidates |

**Aggregate match rate:** ~54% across all timesteps (random baseline: 32.8%, distribution-overlap upper bound: 91.5%).

**Interpretation:** Bull/Bear/Crisis Kronos features are reliably aligned to real regime-matching windows. SevereBear gets degraded signal (often aligned to Bear days), but degraded ≠ corrupted — Bear-day Kronos forecasts are semantically the closest available substitute. This pattern is structural to the rare-regime entry rate of the simulator and is not fixable by alignment tuning alone.

---

## Build provenance

| Component | Source |
|---|---|
| Fitted regime/GARCH/copula parameters | `data/synthetic/models/synthetic_generator_FITTED.pkl` |
| Real Kronos source | `data/proccessed/combined_w_cross_asset/train/RL_Final_Merged_train.csv` |
| Schema reference | Same CSV (318 columns) |
| HMM training window | 2004-02-09 → 2022-12-30 (4,757 trading days) |
| CSV training window | 2004-10-25 → 2022-12-30 (4,579 trading days; 178-day technical indicator buffer trimmed off start) |
| Date alignment | Date-based join (CSV dates → pickle regime labels). All 4,579 CSV dates present in pickle. |

---

## Performance

| Stage | Wall time |
|---|---|
| Pickle + CSV load | < 1s |
| Simulator (`simulate_hybrid_paths`, 2000 paths × 584 steps) | ~6s |
| Kronos match quality sample | < 1s |
| Per-path feature build (technicals + wavelets + cross-asset + Kronos attach) | ~50 min |
| Save (compressed npz) | ~2s |
| **Total** | **51.5 min** |

Per-path build cost dominates total time (~1.5 sec/path).

---

## Companion outputs

| File | Purpose |
|---|---|
| `data/synthetic/pools/synthetic_pool_production_n2000_seed42.npz` | Earlier pool with trim_warmup=128 (0.30% NaN in 28 columns). Valid as a second sample if NaN handling is added downstream. |
| `data/synthetic/pool_checkpoints/checkpoint_seed43_paths*.npz` | Crash-recovery snapshots at every 200 paths. Safe to delete now that full build succeeded. |
| `data/synthetic/pools/synthetic_pool_test_n50_seed42.npz` | 50-path validation pool from earlier test runs |
| `data/synthetic/pools/synthetic_pool_small_demo_n4_seed42.npz` | 4-path warmup-validation pool (NaN-free at trim_warmup=200) |

---

## Loading for RL training

```python
from build_synthetic_pool_v2 import load_pool, PoolSampler
import numpy as np

pool = load_pool('data/synthetic/pools/synthetic_pool_production_n2000_seed43.npz')

print(f"Features:      {pool['features'].shape}")        # (2000, 384, 317)
print(f"Feature names: {len(pool['feature_names'])}")     # 317
print(f"Metadata:      {pool['metadata']}")               # full build config

# Batched sampling
sampler = PoolSampler(pool)
rng = np.random.default_rng(0)
batch = sampler.sample(batch_size=128, rng=rng)
# batch['features']: (128, 384, 317) float32
# batch['returns']:  (128, 384, 4)   float32
# batch['prices']:   (128, 385, 4)   float32
# batch['regimes']:  (128, 384)      int8
# batch['path_idx']: (128,)          int64  — path indices used in batch
```

---

## Known caveats

1. **SevereBear Kronos features are noisier than other regimes.** When attributing RL agent performance, expect slightly weaker signal in SevereBear states. If this becomes a training issue, consider down-weighting Kronos columns conditional on regime, or stratifying the alignment by regime segment.

2. **Crisis is over-represented vs reality.** The simulator generates ~13% Crisis at the 20-path level (averaging out to 7.5% across 2000 paths) due to elevated rare-regime entry rates. This is a feature, not a bug — the agent benefits from stress exposure that 18 years of real history could not provide.

3. **Calendar dates are discarded.** The features tensor uses positional indexing (rows 0..383 = consecutive trading days). If the RL agent needs day-of-week, holiday, or month-end gates, reconstruct dates from `start_date` + business-day offset.

4. **Companion `prices` array has length 385** while `features`/`returns`/`volumes`/`regimes` have length 384. This is intentional — `prices[0]` is the prior close, `prices[t+1]` is the close after return `returns[t]`.
