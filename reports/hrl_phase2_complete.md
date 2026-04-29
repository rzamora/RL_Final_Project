# HRL Phase 2 — Complete Findings: Hierarchy Doesn't Help This Task

## Executive Summary

Across five architectures spanning the reasonable HRL design space, **no hierarchical variant beats the flat LL-alone baseline on real test data**. The flat policy (light_100k LL with fixed HL action `[0.33, 0.5]`) achieves 1.531x equity, Sharpe 1.87, max drawdown 0.112 on real test. Every HRL variant — three frozen-LL configurations and one joint LL+HL training — underperforms this baseline. Joint training, the architecturally principled answer to coordination problems, performs the worst (0.94x equity, Sharpe -0.46 on real test, with the 1M endpoint losing money on a risk-adjusted basis).

The result is overdetermined. The architectural conclusion is no longer "we picked the wrong setup" — it's "this task does not benefit from hierarchical structure as currently formulated." The remaining levers are not architectural: they are the data distribution and the regime feature pipeline.

## What We Built

A two-level HRL system for portfolio allocation across NVDA, AMD, SMH, TLT.

**Low-level (LL) policy** chooses per-asset weights given an HL action. 4-dim action space (one weight signal per asset). Receives 325-dim observation: 313 features + 10 portfolio_state + 2 appended HL action.

**High-level (HL) policy** chooses `[gross_signal, net_signal] ∈ [-1, +1]²` per step. Gross_signal maps to `target_gross = (gross+1)/2 * max_gross` (a leverage knob ranging from 0 to 1.5). Net_signal maps to `target_net = net * target_gross` (a directional knob from fully short to fully long). 2-dim action space, 323-dim observation (no HL action append).

**Reward** is asymmetric across upside/downside × beat/miss benchmark, with rolling-window excess drawdown penalties. Heavy weight (0.5) on upside_miss (career risk for missing rallies) and crisis_alpha (the prize). Mild weight (0.1) on upside_beat and downside_excess. Plus log-growth and turnover terms.

**Data**:
- Synth pool: 2000 paths × 384 days × 313 features. Per-step regime distribution: Bull 39%, Bear 36%, SevereBear 17%, Crisis 7%.
- Real train: 4579 days, 2004-2022.
- Real test: 829 days, 2023-2026. Per-day regime distribution: Bull 31%, Bear 30%, **SevereBear 38%**, Crisis 1%.

The mismatch between synth and real-test regime distributions is significant — synth is Bull-heavy, real test is SB-heavy. This is one of the suspected weak points.

## Five Architectures Tested

### Phase 1 baseline: LL-only with fixed HL action

LL trained with `[gross_signal, net_signal] = [0.33, 0.5]` constant for the entire training run. Synth pretrain (1M steps) + light real-data fine-tune (100k steps). The HL "decision" is fixed by hand at training time.

**Real test result**: 1.531x equity, alpha -0.172, Sharpe +1.87, Sortino +3.09, max_dd 0.112. **This is the best result of any architecture.** Strongly negative alpha because the EW benchmark on test is essentially leveraged-long during 2023-2026 bull, and the agent is risk-managed; but the agent itself returns a competitive 1.53x with low drawdown.

A critical finding from the per-regime diagnostic on this LL: the agent is **regime-blind**. NVDA weight 0.17 in Bull / 0.17 in Bear / 0.11 in SB / 0.07 in Crisis. TLT weight 0.28 / 0.31 / 0.31 / 0.28. The differences across regimes are small. The 1.87 Sharpe is achieved by a near-fixed defensive allocation that benefits from TLT diversification, not by tactical regime-aware behavior. This was the motivation for Phase 2 HL training.

### v1: Frozen LL = light_100k

HL trained on synth pool with frozen LL = light_100k. The LL was never exposed to varied HL actions during training, so its response to anything other than `[0.33, 0.5]` is unpredictable. The "Fix A" patch (zero out last 2 dims of LL VecNormalize) attempted to mitigate this.

**Posture diagnostic outcome**: `posture/net_gap_SB_minus_Bull` stayed within ±0.02 across the full 1M-step training. The HL never learned to differentiate posture by regime. Mean net drifted from -0.02 toward +0.20 (gravitating toward the LL's training point), then collapsed at 1M.

**Real test result (1M final)**: 1.059x equity, alpha -0.650, Sharpe +0.35, max_dd 0.097. Underperforms LL alone by 0.47 in equity. Sharpe collapses to nearly random baseline.

**Reason for failure**: the frozen LL effectively penalizes the HL for trying anything other than `[0.33, 0.5]` because the LL's response to off-distribution HL actions is unreliable. The HL gets a noisy gradient signal that drowns out the regime structure it should learn.

### v2: Frozen LL = ll_random_hl_finetune (generalist)

LL retrained on synth + real_train with HL action sampled `Uniform(-1, +1)²` per episode. Designed to handle any HL action gracefully. Per-HL-action diagnostic confirmed the LL responds correctly to varied HL actions: `[+0.7, +0.7]` produces 1.77x equity, `[+0.5, -0.5]` produces 0.95x (correctly losing on shorts during a bull), `[-0.9, 0.0]` produces 1.03x (near-flat as instructed).

**Posture diagnostic outcome**: `net_gap_SB_minus_Bull` reached -0.029 at 1M (slightly negative, meaning SB is more defensive than Bull — the desired direction!) but only -0.029 in magnitude. The HL did learn regime differentiation, just very weakly. Mean net drifted from -0.02 to +0.37 over training — the HL learned "go more long" globally rather than regime-conditional.

**Real test result (1M final)**: 1.217x equity, alpha -0.497, Sharpe +0.67, max_dd 0.237. Better than v1 but still underperforms LL alone by 0.31 in equity. The 0.237 max_dd is the worst we've seen — the HL adds risk without adding return.

**Reason for failure**: the LL is genuinely robust across the HL action space (good), but spends capacity on combinations that are objectively wrong (Bull regime + short net). The HL's gradient signal exists but the LL is reasonable everywhere, so the gradient doesn't pull the HL strongly toward regime-conditional behavior. The HL wanders, finds "be more long globally" as a reasonable default, and ends up with risk-on exposure that gets hurt during test SB days.

### v3: Frozen LL = ll_regime_bucket_hl_finetune (specialist)

LL retrained with HL action constrained to disjoint regime buckets:
- Bull bucket: `[+0.5, +1.0] × [+0.5, +1.0]`
- Bear bucket: `[+0.0, +0.5] × [+0.0, +0.5]`  
- SevereBear bucket: `[-0.5, +0.0] × [-0.5, +0.0]`
- Crisis bucket: `[-1.0, -0.5] × [-1.0, -0.5]`

Modal regime per episode determines the bucket. Designed to maximize gradient signal at bucket centers — when the HL produces a Bull-bucket action during a Bull regime, the LL has been extensively trained on that combination and produces strong regime-appropriate weights. The hope: PPO gradient pulls HL output toward bucket centers.

Per-bucket diagnostic on real test confirmed the LL has clean regime-conditional structure: Bull center → 1.524x equity, Bear center → 1.191x, SB center → 0.975x (defensive, slight loss), Crisis center → 0.949x (most defensive, biggest loss on a bull test). Spread of 0.575 in equity across regimes.

**Posture diagnostic outcome**: `net_gap_SB_minus_Bull` reached +0.111 at 700k (the largest magnitude of any version), but in the *wrong* direction — SB had higher net than Bull. The HL learned that during synth SB regimes, going long produces better rewards (because synth SB regimes contain mean-reversion bottoms). The bucket structure didn't anchor the HL to the right regime corners.

`bucket/correct_fraction_overall` started at 0.41 (random init coincidentally fell near SB bucket) and dropped to 0.04 by 1M. The HL learned to operate in the *gaps between buckets* where the LL produces near-flat, low-conviction positions — the "safe" zone where small mistakes happen instead of large mistakes.

**Real test result (1M final)**: 0.977x equity, alpha -0.720, Sharpe +0.08, max_dd 0.250. The first variant to *underperform random baseline* on equity. Worst max_dd of any HRL variant.

**Reason for failure**: the reward landscape between buckets isn't flat — it's *safer* than the buckets themselves. PPO found local optima in the in-between zones. The structural argument for bucket conditioning (clear gradient signal at bucket centers) failed because the gradient doesn't actually pull there.

### Joint LL+HL training

Single PPO model with structured policy:
- Shared LayerNorm trunk: 325-dim obs → 256-dim latent
- HL head: latent → 2-dim Diagonal Gaussian
- LL head: `[latent, HL_sample]` → 4-dim Diagonal Gaussian (B1 — sampled HL passed to LL)
- Value head: latent → scalar V(s)
- Joint distribution log_prob = HL log_prob + LL log_prob; entropy and gradient flow through both via reparameterization

The architecturally cleanest answer to the frozen-LL coordination problem. Both heads update together end-to-end. No frozen anything. Backprop flows from reward → LL action → HL sample (via rsample reparam) → HL parameters.

**Posture diagnostic outcome**: `net_gap_SB_minus_Bull` stayed within ±0.02 throughout the entire 1M-step run. Joint training did not learn regime-conditional posture either. Mean net drifted from +0.025 → -0.040 → 0.000, and gross_overall_mean drifted from 0.75 → 0.48. **The joint policy learned global defensiveness**, not regime-conditional behavior.

KL behaved cleanly (stayed under 0.035). Std remained stable at 1.0. clip_fraction climbed but didn't blow up. Mechanically the training was healthier than v1/v2/v3.

**Real test result (1M final)**: 0.941x equity, alpha -0.751, Sharpe **-0.46**, max_dd 0.174. **The 1M endpoint loses money on a risk-adjusted basis on a bull market.** Worst HRL variant of all.

**Reason for failure**: the joint architecture removed the coordination problem (HL and LL update together, no train-deploy mismatch), but the resulting policy is *more* defensive than the frozen-LL variants. With no frozen LL forcing a particular posture, the joint optimizer is free to settle on "near-flat positions" because that's what minimizes downside risk. Combined with a strong-bull test period, defensiveness loses.

## Headline Comparison Table

All numbers on real_test, 10-seed median.

| variant | eq | alpha | Sharpe | Sortino | Calmar | max_dd | hit |
|---------|-----|-------|--------|---------|--------|--------|-----|
| **light_100k LL alone (FLAT)** | **1.531** | **-0.172** | **+1.87** | **+3.09** | **+2.96** | 0.112 | 0.546 |
| synth_600k LL alone | 1.434 | -0.263 | +1.61 | +2.60 | +2.46 | 0.110 | 0.532 |
| hl_v1_300k_best | 1.223 | -0.478 | +1.60 | +2.49 | +2.96 | **0.053** | 0.540 |
| hl_v2_1M_final | 1.217 | -0.497 | +0.67 | +0.84 | +0.66 | 0.237 | 0.504 |
| hl_v2_900k_best | 1.143 | -0.557 | +0.62 | +1.17 | +1.11 | 0.085 | 0.505 |
| random | 1.124 | -0.577 | +0.45 | +0.68 | — | 0.232 | 0.500 |
| hl_v3_750k_best | 1.094 | -0.621 | +0.51 | +0.76 | +0.57 | 0.128 | 0.504 |
| hl_v1_1M_final | 1.059 | -0.650 | +0.35 | +0.42 | +0.41 | 0.097 | 0.395 |
| hl_v3_1M_final | 0.977 | -0.720 | +0.08 | +0.09 | -0.06 | 0.250 | 0.432 |
| joint_200k_best_train | 0.942 | -0.728 | -0.21 | -0.23 | -0.20 | 0.184 | 0.512 |
| joint_1M_final | 0.941 | -0.751 | -0.46 | -0.59 | -0.30 | 0.174 | 0.509 |
| **joint_50k_best_test** | **0.925** | **-0.759** | **-0.32** | -0.36 | -0.22 | 0.201 | 0.507 |

The flat policy beats every hierarchical variant. Joint training, despite being architecturally cleanest, performs worst.

## Posture Diagnostic Across Variants

The single most important diagnostic for "did the HL learn regime-conditional posture" was `posture/net_gap_SB_minus_Bull` — the difference in mean net exposure between SB and Bull regimes on real test. We expected this to become significantly negative (HL goes more defensive in SB) for a successful HRL system.

| variant | gap @ 200k | gap @ 500k | gap @ 1M | net@SB - net@Bull at 1M |
|---------|-----------|-----------|----------|--------------------------|
| v1 | +0.012 | 0.000 | +0.007 | +0.21 vs +0.20 |
| v2 | +0.030 | +0.015 | -0.029 | +0.34 vs +0.26 |
| v3 | -0.001 | +0.003 | +0.087 | +0.34 vs +0.26 |
| joint | +0.013 | +0.017 | +0.012 | -0.001 vs -0.009 |

**No variant produced a meaningfully negative gap.** v2 is the closest at -0.029, but the *overall* mean nets are both still long-biased — the HL is not learning "be defensive in SB"; it's learning "be more long in SB than in Bull, both still long." Joint training collapsed both numbers near zero — the policy is near-flat regardless of regime.

## Why Hierarchy Failed: Three Hypotheses, Ranked

### 1. The regime classifier degrades on real test

Documented in Phase 1: classifier was meaningful on real_train (Bull NVDA +0.35%, SB NVDA -0.046%) but degraded on real_test (Bull NVDA +0.015%, AMD -0.22%, TLT -0.10% during Bull-labeled days). The labels stop predicting equity direction.

The HL is supposed to learn "regime → posture" mapping. If on test the regime labels don't structurally correspond to expected market behavior, no HL can learn a useful mapping. The conditioning signal isn't there to be found.

This is the most likely root cause. Evidence: every architecture failed similarly on test, even when the LL itself produced clean regime-conditional structure (v3). The bottleneck is in the input features, not the policy.

### 2. The reward function rewards global defensiveness on this test period

The asymmetric reward heavily penalizes upside_miss (lambda=0.5) and crisis_alpha (lambda=0.5) but only mildly penalizes downside_excess and upside_beat (lambda=0.1). The intent: encourage participation in rallies and crisis alpha. The unintended consequence: with 2023-2026 test being a strong bull market that *also* contains 38% SB-labeled days, *any* defensive posture gets hit hard during bull days while only mildly rewarded during SB days.

A flat "stay long" policy outperforms a regime-aware policy because the test period rewards staying long more than it rewards correctly identifying SB days. Joint training converged to "near-flat" specifically because that's the safest posture under this reward.

### 3. Synth-test distribution mismatch

Synth modal regime distribution: Bull 48%, Bear 37%, SB 11%, Crisis 5%. Real test per-day distribution: Bull 31%, Bear 30%, **SB 38%**, Crisis 1%. The HL trains predominantly on Bull/Bear-modal synth paths and is then evaluated on a SB-heavy test period.

The HL has limited exposure to "make defensive decisions during prolonged SB regimes" during training, so what it learns about SB doesn't generalize. Synth SB regimes also tend to be short stretches between Bull/Bear segments rather than sustained drawdowns, so the SB content the HL sees is *not* pedagogically useful for "what to do when 38% of test is SB."

## Possible Next Steps (Not Yet Tested)

### A. Reweight synth pool to match test distribution

Cheapest intervention. Modify `SyntheticPoolCoreSampler` to draw paths with weights aligned to real-test regime composition (e.g. 25% Bull-modal / 25% Bear-modal / 35% SB-modal / 15% Crisis-modal). The Crisis weight is boosted above test (1%) because at 1% the regime essentially doesn't appear in 384-day test windows; over-representing Crisis during training is the only way to give the agent a chance to learn it.

This tests whether the failure is data-distribution (fixable) or fundamental (regime classifier degradation). If reweighted training improves real_test, we have a story. If not, attention should shift to feature pipeline.

**Cost**: 30 minutes to implement and run. Very high leverage if it works.

### B. Fix the regime classifier degradation

The hardest but highest-leverage intervention. The current regime classifier was trained on real_train data and degrades out-of-sample. Possible improvements:
- Retrain classifier with regularization (cross-validation, dropout, ensemble)
- Use a more conservative classifier (logistic regression on macro variables) for stability
- Add classifier-confidence as an additional feature so the HL can learn to ignore the regime when the classifier is uncertain
- Replace fixed regime labels with continuous regime probabilities and let the HL consume them directly

**Cost**: several days. May require a separate ML pipeline. But would address the fundamental cause.

### C. Reshape the reward function

If we believe the reward is encouraging global defensiveness over regime-conditional behavior, we could:
- Add a regime-aware reward term: bonus for posture matching regime expectations (e.g., Bull posture during Bull regime)
- Increase lambda_crisis_alpha further to make defensive posture during SB more rewarding
- Use a regime-stratified evaluation reward that explicitly weights SB-day and Crisis-day rewards higher

**Cost**: moderate. Risk: introduces explicit prior beliefs (Bull → long), partially defeating the "let the agent learn" RL framing.

### D. Generate targeted synth data for failure cases

Beyond reweighting, generate *new* synth paths that contain the structural conditions the agent fails on:
- Sustained SB regimes (50+ days of consecutive SB labels)
- Sharp Bull → Crisis transitions
- Crisis regimes that *don't* mean-revert (where defensive posture compounds value)

**Cost**: depends on how synth was generated. If from a generative model, regenerate with new conditions. If bootstrap-resampled real data, need historical SB/Crisis periods.

### E. Use a learned signal for HL action vs labeled regime

The current setup gives the HL the regime probabilities as input features. Maybe the HL should produce its own latent regime understanding from raw features rather than relying on a (flawed) labeled classifier. This would mean the HL learns *both* the regime classifier and the posture policy end-to-end. Significant additional capacity required.

**Cost**: largest. Effectively redesigning the HL into a more sophisticated representation learner.

## Recommended Path Forward

In order of effort vs expected information value:

1. **A (reweight synth pool)**. Cheapest, highest leverage if it works. If it doesn't help any architecture, that's strong evidence the problem isn't data distribution. **Priority 1**.

2. **B (fix regime classifier)**. The most important intervention if A fails. Probably the actual root cause but requires the most effort.

3. **D (generate failure-case synth)**. Only worth doing if A shows promise — extends A with more targeted data.

4. **C (reshape reward) and E (learn regime end-to-end)** are larger architectural changes that should only be considered after the data-side interventions are exhausted.

## What This Means for the Project

If reweighting (A) helps, we have a positive story: "HRL works on this task once training data matches deployment distribution." We can demonstrate regime-conditional posture and show the HL actually adds value on test.

If reweighting (A) doesn't help, we have a negative-but-rigorous story: "we tested five architectures plus a data-distribution intervention; the bottleneck is the test-period regime classifier; HRL on this task requires solving the feature problem before the architecture problem." This is a defensible grad-level conclusion.

Either way, the architectural exploration is essentially complete. The next phase is data and features, not policy networks.