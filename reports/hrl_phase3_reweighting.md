# HRL Phase 3 — Synth Pool Reweighting: Mixed Results, Data-Distribution Hypothesis Partially Confirmed

## Executive Summary

We retrained the LL-only baseline using a reweighted synth pool sampler that targets the real-test regime distribution: 25% Bull-modal / 25% Bear-modal / 35% SevereBear-modal / 15% Crisis-modal (vs the natural pool's 48/36/11/5 distribution). The hypothesis: training on a regime-balanced pool would produce a policy that generalizes better to the SB-heavy test period (38% SB days).

The result is **mixed and informative, but does not beat the light_100k baseline**:

- **On real_train**: dramatic improvement. Final equity 1.956x with Sharpe +2.47, hit_rate 0.557, **positive alpha (+0.417)**. The first time any of our policies achieved positive train alpha. Eval reward improved from ~-7 (typical pretrain endpoint) to +2.21.
- **On real_test**: 1.368x equity with Sharpe +0.96, max_dd 0.216. Underperforms light_100k baseline (1.531x, Sharpe +1.87, max_dd 0.112) on every metric.

The reweighting changed what the agent learned — clearly and meaningfully — but the change overfit to the reweighted training distribution rather than improving test generalization. Two distinct mechanisms explain the gap, both pointing to follow-up experiments.

## Setup

### Reweighted sampler

`ReweightedSyntheticPoolCoreSampler` subclasses the original `SyntheticPoolCoreSampler` and overrides `reset()` to use weighted regime-then-path sampling:

1. Classify each of the 2000 pool paths by modal regime (the most-frequent regime label across its 384 days).
2. On reset(), draw a modal regime from the target distribution.
3. Sample a path uniformly from the paths matching that regime.

This means each individual path gets sampled at a rate proportional to its regime weight divided by the number of paths in that regime group.

### Pool composition vs target

| regime | pool natural | n_paths | target | per-path multiplier |
|--------|--------------|---------|--------|---------------------|
| Bull-modal | 47.9% | 958 | 25% | 0.52x |
| Bear-modal | 36.5% | 730 | 25% | 0.69x |
| SevereBear-modal | 10.5% | 211 | 35% | 3.34x |
| Crisis-modal | 5.1% | 101 | 15% | 2.97x |

Over 1M training steps with 384-step episodes (~2600 episodes), each Crisis path gets sampled ~3.9 times on average. Each SB path gets ~4.4 times. Each Bull path gets ~0.7 times. Bear paths get ~0.9 times.

### Training pipeline

Two stages, both with fixed HL action `[0.33, 0.5]` (same as the original light_100k pipeline):

1. **Synth pretrain** (1M steps): reweighted synth sampler, lr=1e-4, n_epochs=4, clip=0.2, ent_coef=0.02. Saved as `synth_reweighted_pretrain_final.zip`.
2. **Real fine-tune** (200k steps): real_train CSV without reweighting, lr=3e-5, n_epochs=4, clip=0.1, ent_coef=0.01. Loads source vecnorm with training=True. Saved as `finetune_reweighted_final.zip`.

Compute time: ~7.5 min synth pretrain + ~1.5 min fine-tune = ~9 min total.

## Training Trajectory

### Synth pretrain (1M steps)

Real-test eval reward improved monotonically across the run, ending at the best value we've seen for any synth pretrain endpoint:

| step | real_train eval | real_test eval |
|------|-----------------|-----------------|
| 200k | -12.99 ± 13.0 | -23.77 ± 9.55 |
| 400k | -11.10 ± 5.0 | -17.59 ± 5.96 |
| 600k | -9.35 ± 8.6 | -20.74 ± 7.21 |
| 800k | -10.49 ± 5.9 | -15.07 ± 4.65 |
| 1M | -12.35 ± 9.6 | **-11.36 ± 1.91** |

Notable: the final real_test eval std is 1.91 — extremely tight compared to other runs (typical std is 5-15). The policy is producing consistent test rewards, which usually indicates a stable converged policy.

### Real-data fine-tune (200k steps)

Real_train eval went positive for the first time across our experiments:

| step | real_train eval | real_test eval |
|------|-----------------|-----------------|
| 50k | -7.47 ± 11.0 | -16.44 ± 3.79 |
| 100k | -6.79 ± 6.5 | -11.77 ± 3.09 |
| 150k | **+1.64 ± 12.4** | -12.20 ± 4.85 |
| 200k | **+2.21 ± 7.8** | -14.74 ± 2.27 |

Real_train alpha became positive between 100k and 150k. Real_test peaked at 100k (-11.77) and degraded slightly through 200k. This is the textbook overfitting signature.

## Portfolio Metrics — Real Test

10-seed median, deterministic policy.

| variant | eq | alpha | Sharpe | Sortino | Calmar | max_dd | hit |
|---------|-----|-------|--------|---------|--------|--------|-----|
| **light_100k LL alone (BASELINE)** | **1.531** | **-0.172** | **+1.87** | **+3.09** | **+2.96** | **0.112** | 0.546 |
| synth_600k LL alone | 1.434 | -0.263 | +1.61 | +2.60 | +2.46 | 0.110 | 0.532 |
| **finetune_reweighted_200k** | 1.368 | -0.386 | +0.96 | +1.32 | +1.09 | 0.216 | 0.509 |
| synth_reweighted_1M (pretrain only) | 1.280 | -0.435 | +0.76 | +1.06 | +0.82 | 0.213 | 0.521 |
| hl_v1_300k_best | 1.223 | -0.478 | +1.60 | +2.49 | +2.96 | 0.053 | 0.540 |
| hl_v2_1M_final | 1.217 | -0.497 | +0.67 | +0.84 | +0.66 | 0.237 | 0.504 |
| random | 1.124 | -0.577 | +0.45 | +0.68 | — | 0.232 | 0.500 |
| hl_v3_1M_final | 0.977 | -0.720 | +0.08 | +0.09 | -0.06 | 0.250 | 0.432 |
| joint_1M_final | 0.941 | -0.751 | -0.46 | -0.59 | -0.30 | 0.174 | 0.509 |

The reweighted variant lands between light_100k baseline and the HRL frozen-LL variants. **It beats every HRL variant** but underperforms both LL-alone baselines (light_100k and synth_600k).

## Portfolio Metrics — Real Train (the surprise)

| variant | eq | alpha | Sharpe | Sortino | max_dd | hit |
|---------|-----|-------|--------|---------|--------|-----|
| **finetune_reweighted_200k** | **1.956** | **+0.417** | **+2.47** | **+4.04** | 0.147 | 0.557 |
| synth_reweighted_1M (pretrain) | 1.247 | -0.305 | +0.75 | +1.15 | 0.186 | 0.514 |

This is unprecedented. **The reweighted fine-tune is the first policy across all our experiments that achieves positive alpha on training data.** Sharpe 2.47 with hit_rate 0.557 indicates a meaningful, consistently profitable train policy. Compare to the light_100k baseline whose train metrics are typically Sharpe ~1.5 with negative alpha against the leveraged-EW benchmark.

The discrepancy between train (eq 1.956, Sharpe 2.47) and test (eq 1.368, Sharpe 0.96) is large. The policy learned aggressive, regime-discriminative behavior on train that didn't transfer to test.

## What Worked

The reweighting **clearly changed what the policy learned**. Multiple lines of evidence:

1. **Real_train eval went positive for the first time.** No prior architecture achieved this. The agent is making profitable allocation decisions on training data.
2. **Final synth pretrain real_test reward (-11.36) is the best synth-only endpoint we've seen**, with unusually tight std (1.91). The policy converged to something stable rather than wandering.
3. **It beats every HRL variant on real_test.** A flat policy with reweighted training data beats hierarchical policies on the original training data.

These results disprove the strongest version of the "data distribution doesn't matter" null hypothesis. Distribution does matter; the agent's learned behavior is responsive to which regimes it sees.

## What Didn't Work

The reweighting **did not beat the light_100k baseline on real_test**. Three diagnostic patterns:

1. **Train-test gap widened.** Light_100k baseline has a small train-test gap (~5-10% in equity). Reweighted has a 30%+ gap (1.96x train vs 1.37x test).
2. **Max drawdown nearly doubled** vs baseline (0.216 vs 0.112). The agent takes more risk and the test period punishes it.
3. **Hit rate dropped slightly** (0.509 vs 0.546). Per-day decisions are less consistently profitable.

The reweighted policy looks more aggressive on train and that aggressiveness doesn't generalize. This is overfitting in the classical sense: the policy learned the training distribution well, including its idiosyncrasies, and the test distribution differs in the wrong ways.

## Why It Didn't Generalize: Two Mechanisms

### Mechanism 1: Sample re-use overfitting

With reweighting, the 101 Crisis paths get sampled ~3.9 times on average and the 211 SB paths get sampled ~4.4 times. The 2000-path pool appears, to the agent, as roughly:

- ~700 distinct Bull-modal episodes (most paths seen <1 time, missing ~30% entirely)
- ~640 distinct Bear-modal episodes
- ~930 SB-modal episodes (each path seen ~4.4x)
- ~390 Crisis-modal episodes (each path seen ~3.9x)

The agent learns Crisis behavior on **101 distinct Crisis scenarios seen 3.9x each**, not on 391 distinct Crisis scenarios seen once each. Repeated exposure to the same scenarios encourages memorization, not generalization. When real test contains a different Crisis (or in this case, almost no Crisis at all), the agent's Crisis-specific learning doesn't transfer.

This explains the train-test gap: the agent learned what works on the *specific* SB and Crisis paths in the pool, not what works on SB and Crisis regimes generally.

### Mechanism 2: The regime classifier degradation problem

Documented in Phase 1: regime labels on real_test do not predict equity direction the same way they do on real_train. Bull-labeled days on test don't reliably correspond to bull market behavior; SB-labeled days on test don't reliably correspond to defensive opportunities.

If the agent learns "be defensive when label is SB" on training data (where the label means something), and then receives a sequence of mostly-SB labels on test (where the label means less), it produces defensive postures on a market that's actually trending upward. This generates the larger drawdowns and worse hit rates we observe.

The reweighting amplified this problem rather than solving it. By oversampling SB/Crisis training paths, we made the agent more dependent on the regime label as a signal — exactly the wrong direction if the label is unreliable on test.

## Diagnostic Implications

The reweighting result is consistent with **both** "data is the bottleneck" *and* "regime classifier is the bottleneck." We can't distinguish them from this experiment alone:

- **If data is the bottleneck**: more SB/Crisis paths (genuinely new ones, not re-sampled) should fix it. Generate more synth data with stress bias and retest.
- **If the regime classifier is the bottleneck**: no amount of data fixes it. The agent will continue to overfit to whatever regime structure is in the training data because the test regime structure is unreliable.

The empirical signature differs:
- Data-bottleneck diagnosis: more SB/Crisis paths → better real_test, smaller train-test gap.
- Classifier-bottleneck diagnosis: more SB/Crisis paths → similar real_train, no improvement on real_test.

The next experiment can resolve this.

## Next Steps

### Step A — Diagnostic on regime classifier reliability (cheapest, ~10 min)

Before running another training experiment, directly test whether regime labels predict returns on test. Pseudocode:
for each regime r in {Bull, Bear, SB, Crisis}:
test_days_with_regime_r = test_df[test_df.regime == r]
mean_return_per_asset = test_days_with_regime_r[return_cols].mean()
print(f"Regime {r}: mean asset returns = {mean_return_per_asset}")

Compare with the same on training data. If train-Bull has mean NVDA return +0.30% but test-Bull has mean NVDA return +0.05%, the label doesn't carry the same predictive structure. This is a 10-minute analysis with high diagnostic value.

If the classifier is clearly broken on test, no further training experiments will help. If the classifier is reasonable on test, the data-bottleneck story is more likely and Step B is worth running.

### Step B — Regenerate synth pool with stress_bias (medium, ~30-60 min)

Use the existing `build_pool` function with:
- `n_paths = 4000` (double the current pool)
- `stress_bias = {2: 1.5, 3: 1.5}` (boost SB and Crisis transition probabilities by 50%)

Expected output: a pool with ~30-40% SB-modal paths, ~15% Crisis-modal paths, plus more diverse instances of each. Then retrain the LL-only pipeline (with no additional reweighting needed since the natural distribution is now closer to target).

This addresses Mechanism 1 directly. If real_test improves, data-quantity was the issue. If real_test stays at ~1.37, the classifier is the issue.

### Step C — Rerun reweighted training with the regenerated pool (small, ~10 min)

After Step B, also try the reweighting *combined with* the regenerated pool. Target distribution: 25/25/35/15 against a pool that natively has more SB/Crisis content. This tests whether reweighting + larger SB/Crisis pool together produces better generalization.

### Step D — Investigate the reward function (medium, ~1 hour analysis)

The reward function has lambda values: upside_miss=0.5, upside_beat=0.1, downside_excess=0.1, crisis_alpha=0.5. This 5x asymmetry between miss and beat could be incentivizing global defensiveness. Possible adjustments:

- Reduce upside_miss to 0.3
- Increase crisis_alpha to 1.0 (the crisis bonus should be larger to compensate for rare crisis events)
- Add a regime-conditional consistency reward (small bonus for posture matching expected regime behavior)

A reward-tuning experiment would test whether the reward function is shaping behavior in counterproductive ways. This is a different bottleneck from data and would likely help all variants (LL-alone, HRL, joint).

### Step E — Accept the negative result and write up (small, immediate)

Defensible conclusion: "We tested 5 architectures (flat LL, three frozen-LL HRL variants, joint LL+HL training) and 1 data intervention (regime-reweighted synth sampling). Across all 6 experiments, the simple flat LL baseline trained on the unmodified synth pool produces the best real_test result. This is a strong negative architectural finding combined with a partially-confirmed data-distribution effect. The probable root cause is regime classifier degradation on out-of-sample test data, which no amount of architectural sophistication or training-distribution adjustment can fix."

This is a complete, defensible report-level finding right now.

## Recommendation

Do **Step A first** (regime classifier diagnostic). It's 10 minutes of analysis with high information value:

- If the classifier is clearly broken on test → write up Step E (the negative result is the conclusion). Steps B-D would not change the outcome.
- If the classifier looks reasonable → Step B (regenerate pool) and Step C (combined). Then likely Step D (reward function) regardless.

The current state is good for a project conclusion. We have:
- 5 architectures tested (clean negative architectural result)
- 1 data intervention tested (mixed result — train improves, test doesn't)
- Clear hypothesis about why (regime classifier vs sample re-use)
- A 10-minute experiment that can resolve the open question

## Updated Architecture Comparison (Final)

Real_test, 10-seed median, deterministic policy.

| Tier | Variants | Best | Worst |
|------|----------|------|-------|
| Best (LL-alone) | light_100k, synth_600k | eq 1.531, Sharpe 1.87 | eq 1.434, Sharpe 1.61 |
| Middle (data intervention) | reweighted finetune, reweighted pretrain | eq 1.368, Sharpe 0.96 | eq 1.280, Sharpe 0.76 |
| Frozen-LL HRL | v1, v2, v3 (best ckpts) | eq 1.223, Sharpe 1.60 | eq 0.977, Sharpe 0.08 |
| Random | random | eq 1.124, Sharpe 0.45 | — |
| Joint | joint_1M_final | eq 0.941, Sharpe -0.46 | eq 0.925, Sharpe -0.32 |

The flat policy on the natural pool wins. Adding hierarchy hurts. Adding data reweighting helps a bit but not enough.
