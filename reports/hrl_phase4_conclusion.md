# HRL Phase 4 — Final Conclusion: Labeling Bug, Architectural Result Survives, and a Different Failure Mode

## Executive Summary

A regime-classifier diagnostic on real_test data revealed an apparent perfect inversion of the train→test regime→return mapping. Investigation of the labeling pipeline traced this to a column-permutation bug in `predict_test_regime_probs`: the test inference script wrote raw HMM state probabilities into columns named `Bull/Bear/SevereBear/Crisis` without applying the canonical state→name remap that was computed and discarded inside `fit_hmm_regimes`. The fix saves the remap (`state_order`) into the HMM pickle and applies it on test inference.

After fixing the bug, the train/test regime alignment improves dramatically. The empirical Spearman rank correlation between train and test regime→EQ_avg orderings goes from −0.40 to +0.50. Bull-day hit rates climb from 52% to 60–73%. Three of four test years are now positively rank-correlated with train (vs. zero of four before).

Two architectural variants were re-evaluated on the corrected test CSV: `joint_1M_final` (the variant with the most capacity to learn regime-conditional behavior at train time) and `finetune_reweighted_200k_final` (the variant with the most explicit regime structure baked into its training distribution). **Both produced byte-identical metrics on the buggy and corrected test CSVs** — same eq, Sharpe, alpha, max_dd, and hit_rate to three decimal places.

This means the policies' decisions on test were unchanged by the corrected regime probabilities. The Phase 2/3 architectural ranking — flat LL baseline > frozen-LL HRL > joint training — survives the bug fix. The previous Phase 4 draft attributed all prior negative results to label inversion; that interpretation is wrong. The real failure mode is different: **the policies do not extract usable signal from the regime probability channel in their observation vectors**, regardless of whether those probabilities are correctly or incorrectly labeled.

The project conclusion is therefore: HRL on this task does not improve over a regime-blind flat policy, and the reason is not classifier degradation. It is that the regime information channel, in the observation vectors as currently constructed, contributes little or nothing to the agents' decisions.

## Part 1 — The Labeling Bug

### Diagnostic that surfaced it

A pandas-only regime-classifier diagnostic computed per-regime mean returns on train and test. The train ranking by EQ_avg was Bull > Crisis > Bear > SevereBear (the expected order). The test ranking — using the test CSV as it then existed — was Crisis > SevereBear > Bear > Bull. Perfect inversion. The full-test Spearman rank correlation against train was −0.40, with 2026 (the smallest sample) showing ρ = −1.0.

A targeted follow-up on the 7 test days argmax-labeled "Crisis" found that all 7 fell in early February 2023 — a calm period, mean EQ_avg +1.42% — not anything resembling a real crisis. Conversely, every actual stress event in the test period (SVB collapse 2023-03-09, Israel-Hamas yields spike 2023-10-19, vol-mageddon 2024-08-05, DeepSeek 2025-01-27, tariff selloff 2025-04-04) was labeled `SevereBear` with probability ≥ 0.98. The 15 largest equity rallies on test were 12-of-15 labeled SevereBear, including +19.9% on 2025-04-09. None of these patterns is consistent with correct labeling.

### Tracing the cause

The HMM is fit once on train data inside `fit_hmm_regimes` (in `regime_dcc_garch_copula_V1.py`). Raw HMM state indices are arbitrary, set by hmmlearn's EM initialization. The function calls `classify_regime_labels` to compute a remap from raw indices to canonical {0:Bull, 1:Bear, 2:SevereBear, 3:Crisis} based on per-state emission means and covariances:

- Bull = state with highest equity mean
- Crisis = state with highest equity volatility among non-Bull states
- SevereBear = state with highest (vol − mean) among the remainder
- Bear = the leftover state

The remap is applied to train probabilities before they are written to `regime_labels.csv`, which is the file consumed by the train side of `merge_state_features.py`. So train labels are canonicalized.

The HMM model and a static name dictionary are saved to a pickle. The remap itself was not saved. At test inference time, `predict_test_regime_probs` loaded the pickle, ran `model.predict_proba` on test features in a strictly causal online loop (no look-ahead), and wrote the resulting probability columns into the test CSV with names `regime_prob_Bull`, etc. — but in raw HMM state order. Since the canonical remap was not available at test time, the columns in the test CSV did not correspond to the names they were given.

For this run, the actual remap was `state_order = [1, 3, 0, 2]`, meaning:

- The column named `regime_prob_Bull` contained probabilities of raw HMM state 0, which canonically is SevereBear.
- The column named `regime_prob_Bear` contained raw state 1, which canonically is Bull.
- The column named `regime_prob_SevereBear` contained raw state 2, which canonically is Crisis.
- The column named `regime_prob_Crisis` contained raw state 3, which canonically is Bear.

A four-cycle permutation. This explains every empirical observation: the calm Feb 2023 cluster argmax-labeled "Crisis" was actually high-probability Bear in HMM space; the SVB/DeepSeek/tariff selloffs argmax-labeled "SevereBear" were actually high-probability Crisis; and the test EQ_avg means by argmax-label were the train EQ_avg means run through the inverse permutation, producing the appearance of perfect inversion.

### The fix

Two lines in `regime_dcc_garch_copula_V1.py`'s `fit_hmm_regimes` save the remap:

```python
with open(DATA_SYN_MOD / 'hmm_4regime.pkl', 'wb') as f:
    pickle.dump({
        'hmm': model,
        'feature_window': 21,
        'regime_label_map': REGIME_NAMES,
        'state_order': order,        # <-- ADDED
    }, f)
```

And in `predict_test_regime_probs`, after collecting raw probabilities from the causal online loop, reorder columns before writing:

```python
state_order = saved['state_order']
probs_canonical = probs_online[:, state_order]
test[prob_columns] = probs_canonical
```

After re-fitting the HMM (so the new pickle contains `state_order`) and re-running `predict_test_regime_probs`, the diagnostic-printed mapping confirmed the bug:

```
raw HMM state 1  ->  canonical 0 (Bull)
raw HMM state 3  ->  canonical 1 (Bear)
raw HMM state 0  ->  canonical 2 (SevereBear)
raw HMM state 2  ->  canonical 3 (Crisis)
```

A non-identity permutation. The bug was real.

## Part 2 — Verification After the Fix

### Per-regime mean returns, corrected

| regime | train EQ_avg | test EQ_avg (buggy) | test EQ_avg (corrected) |
|--------|---------------|----------------------|--------------------------|
| Bull | +0.295% | −0.085% | **+0.458%** |
| Bear | −0.032% | +0.365% | −0.162% |
| SevereBear | −0.075% | +0.462% | +0.389% |
| Crisis | +0.115% | +1.420% | (no test days) |

Bull is now correctly aligned: positive mean equity returns on test (+0.458%) with a magnitude even larger than train (+0.295%). Bear is approximately aligned with the same sign as train.

### Per-year regime alignment, corrected

Spearman rank correlation between train and test EQ_avg orderings, by year:

| year | buggy ρ | corrected ρ |
|------|---------|-------------|
| 2023 | 0.0 | −0.5 |
| 2024 | −0.5 | +0.5 |
| 2025 | −0.5 | +0.5 |
| 2026 | −1.0 | +0.5 |
| ALL | −0.4 | **+0.5** |

Three of four test years now align positively with train. 2023 is the exception, dragged into negative correlation by 2023's high-vol AI-rally early period.

### A residual OOD effect on SevereBear

Even after the fix, SevereBear-labeled test days have positive mean equity returns (+0.389%), opposite to the train signal (−0.075%). This is not a labeling bug — it is a real distribution shift. The HMM's SB state was trained on 2004–2022 data dominated by GFC and 2018-Q4 selloffs (high-volatility, negative-drift episodes). On 2023–2026 test, the high-volatility days that match SB-style emission distributions are the AI-rally's intraday reversals and vol-mageddon-style events, which tended to be followed by sharp recoveries. The classifier identifies the volatility correctly; the realized returns differ in sign because the post-2022 market regime is structurally different.

This OOD effect is real but mild compared to what the buggy labels suggested. Bull and Bear regimes transfer reasonably well; SB transfers poorly; Crisis is empty on test, possibly because no GFC-scale event occurred in the test period.

## Part 3 — The Real Failure Mode

### The decisive re-evaluations

The corrected test CSV differs from the buggy test CSV only in the four `regime_prob_*` columns. All other features (asset returns, prices, volumes, technical indicators, Kronos forecasts, cross-asset correlations) are byte-identical. So any policy whose actions on test depend on regime probabilities should produce different metrics on the buggy vs. corrected CSV. Any policy whose actions do not depend on regime probabilities should produce identical metrics.

Two re-evaluations were run on the corrected test CSV. The variants chosen were the ones most likely to show divergence under the two competing hypotheses about the bug's effect.

**Joint LL+HL training.** All three checkpoints (50k, 200k, 1M) were re-evaluated. The 1M-step final policy is the one previously reported.

| variant | eq (buggy) | eq (corrected) | Δ |
|---------|------------|-----------------|----|
| joint_1M_final @ real_test | 0.941 | 0.941 | 0.000 |

Sharpe (−0.46), alpha (−0.751), max_dd (0.174), hit_rate (0.509) — all identical to three decimal places.

**Reweighted LL fine-tune.** Both checkpoints were re-evaluated.

| variant | eq (buggy) | eq (corrected) | Δ |
|---------|------------|-----------------|----|
| synth_reweighted_1M @ real_test | 1.280 | 1.280 | 0.000 |
| finetune_reweighted_200k @ real_test | 1.368 | 1.368 | 0.000 |

All metrics identical. The first-ever positive train alpha (+0.417, eq 1.956x, Sharpe 2.47) is also reproduced exactly, as expected since train was not affected by the bug.

### What this means

The regime probability columns differ between buggy and corrected test CSVs by a four-cycle permutation. This is a maximum-information change to those four columns — every test day has different values in `regime_prob_Bull` vs. before. If a policy weights those columns nonzero in its decision function, its actions should change.

For both joint and reweighted variants, the actions did not change. The policies' decisions are functionally independent of the regime probability channel.

This is consistent across the two variants chosen. Joint had the most architectural opportunity to learn regime-conditional behavior — both networks training simultaneously, gradient flowing through every observation channel. Reweighted had the most explicit training-time exposure to regime structure — its synth pool sampling distribution was deliberately shifted to over-represent SB and Crisis paths, with the goal of producing a policy that handles those regimes better. Despite very different mechanisms, both produced policies that ignore the regime probability channel at decision time.

### Why no further re-evaluations are needed

The Phase 2 frozen-LL HRL variants (v1 fixed-action LL, v2 random-HL LL, v3 bucket LL) all use the same observation construction as joint and reweighted, with less explicit regime emphasis at training time. If joint (most architectural capacity) and reweighted (most training-time emphasis) both ignore the regime channel, v1/v2/v3 will too. Re-running them would be confirmatory rather than informative. The flat LL baselines (light_100k, synth_600k) are regime-blind by construction; their numbers are unchanged trivially.

The architectural ranking from Phase 2/3 is therefore preserved in its entirety on the corrected test CSV.

## Part 4 — Revised Architectural Conclusion

The previous draft of Phase 4 framed the architectural failures as a consequence of regime label inversion: "every variant that incorporates regime labels into its policy is harmed in proportion to how strongly it uses them." That framing is empirically wrong. The variants do not use the regime labels at decision time, regardless of whether the labels are correctly or incorrectly aligned to market behavior.

The correct framing is:

> Across five architectures (flat LL, three frozen-LL HRL variants, joint LL+HL training) and one data intervention (regime-reweighted synth sampling), the simple regime-blind flat LL baseline produces the best test result. The architectural ranking is preserved when the test labels are corrected. The observed outcome is not a consequence of the regime classifier failing to transfer; it is a consequence of the policies failing to extract usable signal from the regime probability channel in their observation vectors. Both finding-most-likely-to-confirm-label-dependence (reweighted) and finding-most-likely-to-confirm-architectural-coupling (joint) produced byte-identical metrics on buggy vs. corrected labels, supporting this interpretation.

### Why the policies ignore the regime channel — likely candidates

The diagnostic does not reach inside the trained policies to confirm a mechanism. Several plausible explanations exist; this project did not run experiments to discriminate among them.

1. **Channel imbalance.** The observation vector is ~314 dimensions wide. The four regime probability columns are 1.3% of that. Without architectural choices that emphasize regime channels (regime-conditioned subnets, gating, attention), gradient signal flowing through 4 of 314 inputs is small relative to signal through asset-feature inputs. The policy network may learn to weight regime channels near zero simply because they contribute marginally during training.

2. **Information redundancy.** The regime probabilities are computed deterministically from the same asset returns the policy already observes. Any function the policy can extract from the regime probabilities, it can also extract from the raw inputs they were computed from. The HMM compresses information; a policy with access to the uncompressed inputs has no reason to prefer the compressed version.

3. **Training-time vs. observation-time decoupling.** Reweighting changes *what* the policy learns by changing the distribution of training scenarios. It does not force the policy to *use* the regime probability columns in its decision function. A policy can learn "be more defensive during noisy market segments" without that decision being mediated by the regime probability vector — the underlying market features are sufficient.

4. **The regime probabilities are already partially encoded in other features.** The Kronos features, cross-asset correlations, and per-asset wavelet/regime labels in the observation vector contain related information. The marginal contribution of the four `regime_prob_*` columns above what other channels already provide may be near zero.

These are hypotheses. Discriminating among them would require either (a) ablation experiments removing the regime channel from observation vectors entirely (predicted result: identical metrics), (b) gradient-norm probing of the trained policy heads to measure decision-function dependence on each input channel, or (c) architectural experiments with explicit regime-conditioned components. None are within the current project's scope.

## Part 5 — Final Architecture Comparison

Real_test, 10-seed median, deterministic policy. Test CSV is the corrected version. Variants marked "(verified)" had their re-evaluation explicitly confirmed; others have unchanged numbers because either they are regime-blind by construction or share observation construction with verified variants and would behave identically by the argument above.

| Tier | Variant | eq | Sharpe | max_dd | Status |
|------|---------|-----|--------|--------|--------|
| Best | light_100k LL alone | 1.531 | +1.87 | 0.112 | unchanged (regime-blind) |
| Best | synth_600k LL alone | 1.434 | +1.61 | 0.110 | unchanged (regime-blind) |
| Middle | finetune_reweighted_200k | 1.368 | +0.96 | 0.216 | unchanged (verified) |
| Middle | synth_reweighted_1M | 1.280 | +0.76 | 0.213 | unchanged (verified) |
| HRL | hl_v1_300k_best | 1.223 | +1.60 | 0.053 | unchanged (inferred) |
| HRL | hl_v2_1M | 1.217 | +0.67 | 0.237 | unchanged (inferred) |
| Random | random | 1.124 | +0.45 | 0.232 | n/a |
| HRL | hl_v3_1M | 0.977 | +0.08 | 0.250 | unchanged (inferred) |
| Joint | joint_1M_final | 0.941 | −0.46 | 0.174 | unchanged (verified) |

The flat LL baseline wins. Adding hierarchy hurts. Adding data reweighting helps somewhat but not enough to beat baseline.

## Part 6 — Methodological Lessons

This project produced two findings worth carrying forward beyond the architectural result.

**Always save the state→name mapping.** Any pipeline that fits a clustering or mixture model and then assigns semantic names to the cluster indices must persist the mapping from raw indices to names. Saving the model alone is not sufficient. The bug in `fit_hmm_regimes` was specifically that the remap was computed locally inside the fit function and discarded after the train output was written. This is a generic failure mode for any mixture-model-with-named-clusters pipeline.

**Eval-time independence is not implied by training-time exposure.** The reweighting result deserves emphasis. The training pipeline was deliberately constructed to make the agent more dependent on regime structure, by oversampling SB and Crisis paths during training. The agent did learn substantially different behavior — its train-set alpha went from negative to +0.417, its train Sharpe to 2.47. But the *channel through which regime structure entered* during training (the path-sampling distribution) was not the *channel through which the trained policy looks for regime information at decision time* (the four regime probability columns in the observation vector). These are dissociable, and on this task they fully dissociated. A reweighted policy "knows about" regime structure in the sense that its weights reflect training on regime-balanced data, but it does not "use" regime information in the sense of making decisions conditioned on the regime probability vector at inference time.

This dissociation is worth flagging because it complicates a common assumption in HRL/regime-conditional RL literature: that exposing an agent to regime-labeled data during training will produce a policy that conditions on regime labels at inference. On this task at least, that assumption fails.

## Files

- `eval/regime_classifier_diagnostic.py` — per-regime mean-return diagnostic that surfaced the inversion.
- `eval/regime_inversion_by_year.py` — per-year breakdown with Spearman ρ.
- `eval/regime_label_sanity_check.py` — smoking-gun analysis (Crisis-day inspection, known stress dates).
- `src/synthetic/regime_dcc_garch_copula_V1.py` — patched to save `state_order` in pickle.
- `src/data/merge_state_features.py` — `predict_test_regime_probs` patched to apply the canonical remap.
- `reports/hrl_phase2_complete.md` — Phase 2 findings.
- `reports/hrl_phase3_reweighting.md` — Phase 3 findings.
- `reports/hrl_phase4_conclusion.md` — this document.
