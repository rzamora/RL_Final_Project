# Portfolio HRL — Review and Recommendations

This is a complete review of your RL training setup, plus the modules and recommendations you asked for. Five files, organized to be read in order:

1. **`01_code_review.md`** — All bugs and issues in `portfolio_hrl_env.py`, ranked by severity. Three critical bugs (feature-dim mismatch, broken LL projection, sleeve mis-scaling) and several smaller issues.

2. **`02_training_strategy.md`** — How to combine the synthetic pool with real data. The recommended approach is two-phase: synthetic pre-train → real fine-tune. Includes alternative strategies and what *not* to do.

3. **`03_historical_first_recommendation.md`** — Direct answer to your question 4: should you train on historical first? **No.** The synthetic-then-real pipeline is the right call. This document explains why with reference to your own README's regime distribution data.

4. **`portfolio_stats.py`** — Runnable module for portfolio statistics and benchmark comparison. Includes the equal-weight (NVDA, AMD, SMH) baseline you asked for, plus 14 other metrics (Sharpe, Sortino, Calmar, max DD, VaR/CVaR, etc.). Drop-in for your evaluation pipeline.

5. **`portfolio_hrl_env_fixed.py`** — Drop-in replacement for `portfolio_hrl_env.py` with the three critical bugs fixed, plus random episode starts, vectorized envs, VecNormalize, and the recommended training pipeline as a single function.

---

## TL;DR by question

### 1. Code review

Three critical bugs in `portfolio_hrl_env.py` that need to be fixed before any training:

- **Feature schema mismatch.** `process_raw_df` drops `*_close` columns, giving 313 features. The synthetic pool has 317 (it keeps closes). The two data sources are not interchangeable.
- **`parse_ll_action` violates the gross/net targets.** The HL command is not actually being respected — the LL projects in two passes that each undo the previous one. Replaced with a clean long/short-book projection.
- **Sleeves are mis-scaled.** Each sleeve is treated as carrying full portfolio gross, then averaged — works at steady state but breaks during transients. Recommendation: drop sleeves for v1 (the turnover penalty does the same job), add them back later if needed.

Plus seven smaller issues, the most impactful being: no random episode starts, no observation normalization, single env (no parallelism), and reward function mismatch with the README (the implementation actually omits the `λ_up` term — this is intentional and correct, the README is more conservative than what was implemented).

### 2. Training strategies

**Recommended: synthetic pre-train (2M steps, 8 envs) → real fine-tune (500k steps, 4 envs, lower LR).**

Synthetic alone misses calibration to real feature distributions and to the actual Kronos signal quality. Real alone overfits to a single 4,579-day path and under-covers SevereBear (which is 37.8% of your test set vs 16.4% of train). The two-phase recipe gets both benefits in the right order.

Always run 5 seeds. Single-seed RL results on financial data are not interpretable.

### 3. Benchmark module

`portfolio_stats.py` does what you asked: equal-weight (NVDA, AMD, SMH) baseline + agent rollout + side-by-side comparison report.

I ran the EW benchmark on the actual test CSV (2023-01-03 to 2026-04-23, 828 days) to give you a concrete number to beat:

```
EW(NVDA, AMD, SMH) on test set, daily-rebalanced:
  Total return:    +634%
  CAGR:             83.4%
  Annualized vol:   40.7%
  Sharpe:            1.70
  Sortino:           2.59
  Max drawdown:    -41.1%
  Calmar:            2.03
```

That's the AI-rally years and the benchmark is heroic. The agent's job in this test window is **not to maximize return** — it's to deliver a similar return with a better drawdown profile, or a similar drawdown with a less volatile path. Don't expect the agent to beat 83% CAGR; expect it to beat 1.70 Sharpe or 41% max DD.

For comparison, on the train CSV (2004-2022, 4,578 days) the same benchmark has Sharpe 0.67 and max DD 79% — so the test period is unusually generous to a long-only equal-weight strategy.

### 4. Historical-first?

**No, don't do real-only as primary training.** Two specific drawbacks:

- **Single-trajectory overfitting.** One history, one sequence, one set of crashes-and-recoveries. The policy memorizes them.
- **Rare-regime under-coverage.** Test set is 37.8% SevereBear, train is only 16.4%. Test set is 0.8% Crisis, train is 6.8%. The agent will be evaluated mostly on a regime it barely trained on.

The synthetic pool addresses both. Use it. The pipeline in `portfolio_hrl_env_fixed.py::train_pipeline` implements the recommended order.

Keep a real-only baseline in your final results table for comparison — it's the right control to show that the synthetic generator does real work.

---

## Order of operations to actually start training

1. Read `01_code_review.md` and apply the fixes (or use `portfolio_hrl_env_fixed.py`).
2. Validate the fixed env on a tiny budget (10k steps, 1 env, single path) to confirm reward signs are sensible and equity doesn't NaN.
3. Run the two-phase pipeline at full budget (2M + 500k) with seed 0. Check that train-CSV in-sample eval beats the benchmark there (Sharpe ~0.67) by some margin.
4. If yes, run seeds 1–4 of the same pipeline. Wait for all 5 to finish.
5. Do a single deterministic test rollout per seed on the test CSV. Report mean ± std of all metrics vs the EW benchmark.

Don't peek at the test set before step 5. That's the discipline that makes the result publishable.
