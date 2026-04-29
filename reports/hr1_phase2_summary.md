# HRL Phase 2 — Frozen-LL Hierarchy: Three Failed Experiments

## TL;DR

Three frozen-LL HL training experiments (v1, v2, v3) all underperformed the LL-alone baseline on real test data. The failures occurred for different mechanical reasons but converged on the same architectural finding: **frozen-LL HRL does not improve over flat policy on this task**. Joint LL+HL training is the next experiment.

## Setup

The hierarchy: HL chooses `[gross_signal, net_signal] ∈ [-1, +1]²` per step → frozen LL chooses per-asset weights given the HL action → env applies allocation, returns reward. Each version differs only in which LL was frozen.

## The Three Frozen LLs

### v1: light_100k (regime-blind LL)
LL trained with HL action fixed at `[0.33, 0.5]` for the entire LL phase. Has no exposure to varied HL actions. Achieves 1.531x equity / Sharpe 1.87 on real_test when used standalone with that fixed action.

### v2: ll_random_hl_finetune (generalist LL)
LL retrained on synth + real_train with HL action sampled uniformly from `[-1, +1]²` per episode. Designed to handle any HL action. Per-HL-action diagnostic on real_test shows the LL responding correctly to varied HL actions:

| HL action | final eq |
|-----------|----------|
| [+0.7, +0.7] | 1.772 |
| [0.33, 0.5] | 1.542 |
| [+0.5, -0.5] | 0.951 |
| [-0.5, +0.5] | 1.193 |

### v3: ll_regime_bucket_hl_finetune 100k (regime-specialized LL)
LL retrained with HL action constrained to disjoint regime buckets (Bull → top-right corner, Crisis → bottom-left, etc). Modal-regime-per-episode determines the bucket. Designed to maximize gradient signal at bucket centers. Per-bucket diagnostic on real_test:

| Bucket | final eq |
|--------|----------|
| Bull center [+0.75, +0.75] | 1.524 |
| Bear center [+0.25, +0.25] | 1.191 |
| SevereBear center [-0.25, -0.25] | 0.975 |
| Crisis center [-0.75, -0.75] | 0.949 |

## HL Training Results

All three HL variants used identical PPO hyperparameters: lr=3e-5, n_epochs=4, clip=0.1, ent_coef=0.01, 1M synth pretrain steps.

### Posture diagnostic — `net_gap_SB_minus_Bull` over training

The headline diagnostic for "did the HL learn regime-conditional posture." Negative gap = HL goes more defensive in SevereBear than Bull (the desired behavior).

| step | v1 | v2 | v3 |
|------|-----|-----|-----|
| 50k | 0.000 | 0.003 | -0.003 |
| 200k | 0.012 | 0.030 | -0.001 |
| 500k | 0.000 | 0.015 | +0.003 |
| 700k | 0.006 | 0.056 | +0.111 |
| 1M | +0.007 | -0.029 | +0.087 |

v1 and v2 stayed near zero throughout. v3 *did* differentiate by 700k — but in the wrong direction (+0.11 means SB is **more long** than Bull). The HL learned that during synth SevereBear paths, going long produces better rewards (likely because synth SB regimes contain mean-reversion bottoms that pay off being aggressive). The bucket structure of the LL did not anchor the HL to Bull-bucket / Crisis-bucket extremes.

### v3-specific: bucket-correct fraction over training

Fraction of HL actions on real test that landed in the regime-correct bucket. Started at 0.41 (random init happened to fall near SevereBear bucket center). Dropped monotonically:

| step | bucket-correct overall |
|------|------------------------|
| 50k | 0.41 |
| 200k | 0.12 |
| 500k | 0.07 |
| 1M | 0.04 |

The HL learned to operate in the *gaps between buckets* where the LL produces near-flat, low-conviction positions. Hypothesis: the in-between zones are "safer" — the LL makes small mistakes rather than the large mistakes that a wrong-bucket action would produce. PPO found local optima in this safe-but-suboptimal region.

## Real_test Portfolio Metrics — All Variants

Aggregated over 10 seeds, median values reported.

| variant | eq | alpha | Sharpe | Sortino | max_dd | hit_rate |
|---------|-----|-------|--------|---------|--------|----------|
| **light_100k LL alone** | **1.531** | **-0.172** | **+1.87** | **+3.09** | 0.112 | 0.546 |
| synth_600k LL alone | 1.434 | -0.263 | +1.61 | +2.60 | 0.110 | 0.532 |
| hl_v1_300k_best | 1.223 | -0.478 | +1.60 | +2.49 | **0.053** | 0.540 |
| hl_v2_1M_final | 1.217 | -0.497 | +0.67 | +0.84 | 0.237 | 0.504 |
| hl_v2_900k_best | 1.143 | -0.557 | +0.62 | +1.17 | 0.085 | 0.505 |
| random | 1.124 | -0.577 | +0.45 | +0.68 | 0.232 | 0.500 |
| hl_v3_750k_best | 1.094 | -0.621 | +0.51 | +0.76 | 0.128 | 0.504 |
| hl_v1_1M_final | 1.059 | -0.650 | +0.35 | +0.42 | 0.097 | 0.395 |
| hl_v3_1M_final | **0.977** | **-0.720** | **+0.08** | +0.09 | 0.250 | 0.432 |

**Every HL variant on real test underperforms the LL-alone baseline.** The best HL is v1_300k at 1.223x equity and Sharpe 1.60 — still well below light_100k's 1.531x and Sharpe 1.87. The worst is v3_1M at 0.977x with negative Sharpe — worse than random. v3 also has the worst max_dd of any HL variant (0.250).

## Diagnosis

Three different LL training distributions produced three different failure modes — but the same architectural conclusion.

**v1 failure:** LL trained on a single HL action. HL exploration into other actions produces unpredictable LL responses. Gradient signal is too noisy. HL converges to mimicking the LL's training point. No regime conditioning emerges.

**v2 failure:** LL trained across `[-1, +1]²` uniformly. The LL becomes a robust generalist but spends capacity on combinations that are objectively wrong (e.g. Bull regime + short net). HL gradient signal exists but doesn't pull strongly toward regime-conditional behavior — the LL handles everything reasonably well, so the HL can wander. Net mean drifts toward "more long" globally without per-regime differentiation.

**v3 failure:** LL trained only on regime-appropriate actions in disjoint corner buckets. LL produces clean per-bucket weights in isolation. But the HL doesn't learn to land in those buckets — it finds local optima in the in-between zones where the LL's response is noisier but safer. The reward landscape between buckets isn't flat enough to push the HL to the right corners.

## What this means for HRL on this task

Frozen-LL hierarchy is a wall on this dataset. The problem is not LL training distribution — three distinct distributions all fail. The problem is structural: **freezing the LL prevents the HL from receiving a coordinated reward signal**. Whatever the LL was trained to do, the HL inherits its failure modes and cannot route around them.

The two LL diagnostics (v2 random, v3 bucket) confirm the LLs themselves are working correctly when given good HL inputs. The breakdown is in the HL+LL coordination, which frozen-LL training cannot improve.

## Next experiment: joint LL+HL training

Both networks train simultaneously. HL produces an HL action, LL produces weights conditioned on that HL action, env produces reward, gradient flows through both. The networks coordinate during training rather than the HL adapting to a fixed LL.

Expected to be more compute-intensive (forward passes through both networks per env step) but is the principled answer to "the LL and HL keep failing to coordinate." If joint training also fails to beat the LL-alone baseline, that's the strongest possible architectural finding: HRL doesn't add value on this task.