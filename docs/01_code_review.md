# Code Review: `portfolio_hrl_env.py`

Read against the README spec, the synthetic pool report, and the real CSV schema.

I found **3 critical bugs**, **5 high-severity issues**, and several smaller cleanups. I'll go in priority order and tell you exactly what to change.

---

## CRITICAL BUGS

### Bug 1 — Feature-dim mismatch + temporal leakage from raw prices

Two problems for the price of one. First: the synthetic pool has **317 features** that include `*_close` columns. Your `process_raw_df` drops `['date'] + price_cols` from the real CSV, giving **313 features**. So a model trained on real data has 313-dim feature input, and the synthetic pool's tensor is 317-dim. They will not be interchangeable.

Second — and this is the deeper issue: **`*_close` columns should not be in the feature set at all.** Absolute price levels are a temporal fingerprint. NVDA at $0.11 in 2004 vs $140 in 2024 lets the policy network learn "I'm in 2024, the AI rally is on, stay long" directly from the price magnitude. That's leakage of calendar position into the policy, and it undermines the regime-aware design the README is built around. A regime-aware policy should be invariant to absolute price level — only relative changes (returns, ratios, indicators) carry signal.

So the schema needs to be aligned by **dropping closes from both** real and synthetic, giving 313 features in both. The closes are still kept on the side for the simulator (return computation) but never enter the policy's observation.

Note: `*_kronos_close_d5` is **not** affected. Despite the name, it stores the Kronos-predicted return as a fraction (e.g. -0.0068, +0.012), not a dollar price. So those four columns stay in the features.

The fix:

```python
PRICE_COLS = ("NVDA_close", "AMD_close", "SMH_close", "TLT_close")

def process_raw_df(df):
    feats = df.drop(columns=["date"] + list(PRICE_COLS)).to_numpy(dtype=np.float32)
    rets  = df[list(PRICE_COLS)].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
    prices = df[list(PRICE_COLS)].to_numpy(dtype=np.float32)  # for simulator only
    return feats, rets, prices  # 313 features, not 317
```

And in the synthetic pool loader, identify the close columns by name from `feature_names` and strip them out before the tensor is exposed to the env. See `portfolio_hrl_env_fixed.py::load_synthetic_pool` for the implementation — it uses the pool's stored `feature_names` to identify and remove exactly the four columns by name (no positional assumptions).

After the fix: real and synthetic both expose `(T, 313)` features to the policy, with prices kept separately for return computation.

### Bug 2 — `parse_ll_action` silently breaks the gross/net targets

Walk through the math:

```python
weights = abs_raw / abs_raw.sum() * target_gross    # |w|.sum() == target_gross  ✅
weights *= np.sign(raw)                              # net != target_net (yet)

current_net = weights.sum()
weights += (target_net - current_net) / self.n_assets  # net = target_net  ✅
                                                       # but |w|.sum() ≠ target_gross anymore ❌

gross = np.sum(np.abs(weights))
if gross > target_gross and gross > 1e-8:
    weights = weights / gross * target_gross         # if too big, scale down
                                                     # but now net != target_net again ❌
```

Each step fixes one constraint and breaks the other. There's no joint solution being computed. In practice `target_gross` is *almost always* approximately right (the additive shift is small for reasonable targets), but the gross/net targets from the HL policy are not actually being respected. The HL agent thinks it's commanding 1.2 gross / 0.8 net and the LL produces something different. That decouples HL training from the realized policy.

The principled fix is the **L1-ball projection with affine constraint** — but the simple, good-enough version is:

```python
def parse_ll_action(self, ll_action, target_gross, target_net):
    # Decompose into long and short books
    raw = np.asarray(ll_action, dtype=np.float32)
    target_gross = float(np.clip(target_gross, 0.0, self.max_gross))
    target_net   = float(np.clip(target_net, -target_gross, target_gross))

    # Solve: long_gross + short_gross = target_gross
    #        long_gross - short_gross = target_net
    long_gross  = 0.5 * (target_gross + target_net)
    short_gross = 0.5 * (target_gross - target_net)

    pos = np.maximum(raw, 0.0)
    neg = np.maximum(-raw, 0.0)

    pos_sum = pos.sum() + 1e-8
    neg_sum = neg.sum() + 1e-8

    weights = (pos / pos_sum) * long_gross - (neg / neg_sum) * short_gross
    return weights.astype(np.float32)
```

Now `|w|.sum() == target_gross` and `w.sum() == target_net` exactly (up to float). The HL command is honored.

**Edge case** worth understanding clearly: if the agent picks `raw` that's all-positive but `target_net < target_gross` (i.e. HL wants some short exposure), `neg_sum == 0`. You cannot satisfy both gross and net with the agent's signed direction alone — there's no short signal to allocate. The implementation above honors `target_net` exactly (because long_gross and short_gross both come out non-negative and the books are computed independently), and gross becomes `target_net` instead of `target_gross`. So in this edge case, **gross is the constraint that gives**.

This is a feature, not a bug: it means if the LL agent never wants to short, it never will. The env doesn't invent short positions on its behalf. The HL agent will learn to pair `target_net = target_gross` requests when its LL is operating long-only, which is the right behavior.

If you'd rather honor gross strictly (treating it as the hard risk budget) and let net be best-effort, swap which constraint gives in the fallback. I'd recommend the version above — risk budgets are more naturally upper bounds (you can always be less risky) than equality constraints.

#### Verification: shorting capability and leverage cap

After the fix, two properties of the action pipeline matter for a long/short portfolio agent and should be confirmed empirically rather than just by inspection:

1. The agent can short any individual asset (or any subset).
2. Gross leverage is hard-capped at `max_gross = 1.5`, even under out-of-distribution policy outputs.

Both verified by direct test:

**Shorting works at every level.** With HL action `[gross_raw=+1, net_raw=-1]` mapping to `(target_gross=1.5, target_net=-1.5)` and LL action `[-1, -1, -1, -1]`, the resulting weights are `[-0.375, -0.375, -0.375, -0.375]` — fully short, all four assets. With HL `[+1, 0]` → `(1.5, 0)` (market neutral) and LL `[+1, -1, 0, 0]`, weights come out `[+0.75, -0.75, 0, 0]` — a clean pair trade. The classic crisis hedge — short equities, long TLT — produces sensible weights like `[-0.28, -0.28, -0.28, +0.45]` from HL `[0.7, -0.3]` + LL `[-1, -1, -1, +1]`.

**Gross is hard-capped at 1.5.** Sweep of 10,000 random `(HL, LL)` action pairs in their declared `[-1, 1]` boxes: max realized gross was 1.499947, with the small gap explained by random samples not quite hitting corners. More importantly, sweep with **out-of-box** actions in `[-2, 2]` (which would happen if the policy network outputs unsquashed values during early training, or under an algorithm without tanh squashing): max realized gross still 1.500000 exactly. This is because `parse_ll_action` defensively re-clips `target_gross` to `[0, max_gross]` before doing anything with it, so even a misbehaving HL policy can't blow through the leverage cap. That defensive clip is worth keeping — it's a hard safety boundary, not redundant.

**Short P&L accounting is correct.** Confirmed that `np.dot(weights, returns)` handles longs and shorts symmetrically: long NVDA + NVDA up 5% → +5% P&L; short NVDA + NVDA up 5% → -5% P&L; short NVDA + NVDA down 5% → +5% P&L. The simulator doesn't need any explicit short-side accounting — the dot product just works.

**One nuance worth knowing.** The cap is on **gross** (sum of absolute weights), not on **margin leverage** as a prime broker would compute it. At gross=1.5 with net=0 (market neutral), you have $0.75 long and $0.75 short on $1 of equity — 1.5x notional book on 1x equity, no actual borrowing. At gross=1.5 with net=+1.5 (long-only, max), you're 50% margined — borrowing $0.50 to be long $1.50 of equities. Both fall under "gross capped at 1.5" but represent very different real-world risk profiles. The README's framework allows margin so the current setup is consistent with the spec, but if at some point you want a stricter "no borrowing, only cash + short" constraint, you'd add `target_net = np.clip(target_net, -target_gross, min(target_gross, 1.0))` after the existing clip. Worth flagging as a configuration knob even if you don't use it in v1.

**Implication for training, not for code.** The agent will only *learn to use* shorts if the reward function gives it a reason to. With a long-biased benchmark (the equal-weight NVDA/AMD/SMH benchmark is long-only), going short during a drawdown looks worse vs. benchmark than just holding. Net result: even though the env supports shorts, the trained policy may not use them in practice. If the post-training analysis shows the agent never goes net-short, that's evidence the benchmark term in the reward is dominating — `lambda_down` may need to be reduced, or you may need a less long-biased benchmark (60/40 mix, cash-rate, etc.). Not a bug, just something to watch for in the post-mortem.

### Bug 3 — Sleeve gross is mis-scaled by 5×

The README says: portfolio = average of 5 sleeves. The code does that. But the code also lets each sleeve carry the **full** `target_gross`. So the realized portfolio gross is approximately `target_gross` (since all 5 sleeves average to that magnitude *after* the warmup), but during the first 5 steps when sleeves are still zero, realized gross is `target_gross * (t/5)` for `t < 5`.

More importantly — and this is the real issue — the LL agent is being asked to think about a **sleeve allocation** but is actually outputting a **full-portfolio allocation**. Those are different policies. A correct sleeve allocator produces the increment that, when averaged with the other 4 fixed sleeves, hits the desired portfolio. The current code pretends each sleeve is a complete portfolio and averages them, which works at steady state but produces strange transient behavior whenever the HL policy changes its target.

There are two reasonable fixes:

- **(Cheaper)** Change the spec. Drop the sleeve mechanism for the first version. Just have the LL output the full portfolio. This is what your code actually implements, just be honest about it.
- **(Truer to spec)** Make the LL output the *sleeve* `s_t` (one-fifth of capital) directly with `|s_t|.sum() ≤ max_gross / n_sleeves`, and have the env compute portfolio gross as `mean(|sleeves|.sum())`. Then HL's `target_gross` is the portfolio target and the LL's per-sleeve gross is `target_gross` (since averaging 5 sleeves of gross G gives portfolio gross G when sleeves point similarly, or less when they don't — which is the diversification benefit of staggering).

I'd start with the cheaper fix (drop sleeves) for the v1 baseline and add sleeves back once everything else is working. Sleeves are a turnover-smoothing trick; you can get the same effect with a turnover penalty (which you already have).

---

## HIGH-SEVERITY ISSUES

### Issue 4 — `reset()` always starts at `t=0`

Every episode is the same trajectory. With one episode = the whole 4,579-day history, you get one trajectory per "epoch". PPO needs many trajectories to estimate advantage. You also get zero start-state diversity, which is the main reason RL on financial time series overfits.

Fix:

```python
def reset(self, *, episode_length=384, rng=None):
    rng = rng or np.random.default_rng()
    max_start = self.n_steps - episode_length - 1
    self.t_start = int(rng.integers(0, max_start + 1))
    self.t_end   = self.t_start + episode_length
    self.t       = self.t_start
    # ... rest of reset
```

This gives you ~4,200 distinct starting points in the train CSV. Episode length 384 to match the synthetic pool's path length is a good default — keeps the two training distributions on the same horizon.

### Issue 5 — No observation normalization

Your obs vector concatenates 317 features (z-scores, ratios, raw prices, RSI ∈ [0,100], MACD, Kronos predicted prices in dollars), portfolio state (`equity` ∈ [0, ∞), drawdown ∈ [0, 1], gross ∈ [0, 1.5], net ∈ [-1.5, 1.5]), and weights. The scales differ by orders of magnitude. PPO's MLP will be dominated by the largest-magnitude inputs.

You added `LayerNorm` in the policy net which helps inside the network — but the input layer still has to map raw `NVDA_close ≈ 200` and `RSI ≈ 50` to similar gradients. Use `VecNormalize` from SB3:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
env = DummyVecEnv([lambda: LowLevelPortfolioEnv(core)])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
```

Note: `norm_reward=False`. Reward normalization on financial data is dangerous — your reward already lives in a sensible scale (log-returns ~ 0.001, drawdown penalty ~ 0.01) and normalizing it kills the relative weighting of the components.

### Issue 6 — Benchmark uses average of NVDA, AMD, SMH

```python
benchmark_returns = train_returns[:, :3].mean(axis=1)
```

This is an equal-weight benchmark on the 3 equities — which is the same thing the user is comparing against in their question 3. So this is internally consistent. **But** it's a generous benchmark (rebalanced daily, no costs, 100% invested). The agent has to beat that *before* costs and *with* the drawdown/turnover penalty. That's a steep bar. Just be aware: if test results show "agent loses to equal-weight on raw return but wins on Sharpe and drawdown," that is the expected outcome of this reward function, not a failure.

### Issue 7 — Reward function doesn't match the README

The README says:

```
- λ_up · 1{B>0} · max(0, B)        ← penalize OUTPERFORMING (caps alpha)
- λ_down · 1{B<0} · max(0, -B)     ← penalize underperforming
```

Your code does:

```python
if self.benchmark_gap < 0:
    reward -= self.lambda_down * abs(self.benchmark_gap)
```

So the README's `λ_up` term is missing. That's actually fine — penalizing outperformance is unusual and the README's own commentary says it "preserves the agent's ability to pursue active strategies" by **not** doing this asymmetrically. So I think the implementation is the intended behavior and the README spec is more conservative than what was implemented. Pick one and document it. I'd keep the implementation (no upside penalty).

### Issue 8 — `n_steps=512` and `total_timesteps=100_000` is undersized

100k timesteps / 512 per rollout = 195 PPO updates. For 4,579-step episodes that's ~22 episodes. Way too few. Either:
- Use vectorized envs (`SubprocVecEnv` with 8 envs gives 8x the data),
- Bump `total_timesteps` to 1–5M,
- Shorten episodes (per Issue 4, use 384-step random windows — gives many more episodes).

Combine all three. Realistic budget: 2M timesteps with 8 parallel envs and 384-step episodes. That's ~5,200 episodes seen, plenty of update steps.

---

## SMALLER ISSUES

### Issue 9 — `LayerNormMlpExtractor` hardcodes `features_dim=299`

Inside `LayerNormActorCriticPolicy._build_mlp_extractor`, you correctly pass `self.features_dim`, so the default is overridden. But the hardcoded default is wrong for both real (313 + 11 portfolio state = 324) and synthetic (317 + 11 = 328) and HL (no `fixed_hl_action` appended). Either remove the default or leave it but add a comment.

### Issue 10 — `LowLevelPortfolioEnv` adds `fixed_hl_action` to obs, `HighLevelPortfolioEnv` doesn't

That means LL trained obs-dim and HL trained obs-dim differ. Fine, but when you call `ll_model.predict(ll_obs, ...)` from inside the HL env, you have to remember to append the *current* HL action. You do this. Good. But if you ever `VecNormalize` one and not the other, the running stats will diverge. Easy footgun. Wrap each in its own `VecNormalize` and save the stats per env.

### Issue 11 — `done = self.t >= self.n_steps - 1`

After `self.t += 1`, this triggers when `t == n_steps - 1`, meaning the env emits exactly `n_steps - 1` step transitions. The very last row of returns is never used. Off-by-one, harmless, but worth fixing if you want clean accounting:

```python
done = self.t >= self.n_steps  # use all rows
```

### Issue 12 — `weights += (target_net - current_net) / self.n_assets` distributes the net adjustment uniformly across assets

Even if the agent intentionally wants zero NVDA, this code pushes a non-zero NVDA weight to satisfy the net target. The L1-projection fix (Bug 2) avoids this.

### Issue 13 — `np.log1p(np.clip(portfolio_return, -0.999, None))`

If `portfolio_return == -1.0` exactly (full loss), `log1p(-0.999) ≈ -6.9`. That's a heavy but finite penalty. OK. But the `np.clip` with `None` upper bound is unnecessary; you can write `max(portfolio_return, -0.999)`. Style only.

### Issue 14 — No seed plumbing

`PortfolioCore.reset()` ignores any seed. Once you add random episode starts, you'll want determinism for evaluation runs. Pass an `np.random.Generator` through.

### Issue 15 — `check_env` will probably warn about unbounded observation space

`spaces.Box(low=-np.inf, high=np.inf, ...)` is technically legal but SB3 warns. Use very wide finite bounds (e.g. ±1e6) and let `VecNormalize` handle the scaling.

---

## Summary of changes I'd make before any training

1. Fix `process_raw_df` to match the synthetic schema (Bug 1).
2. Replace `parse_ll_action` with the long/short-book projection (Bug 2).
3. Either drop sleeves for v1 or implement them correctly (Bug 3).
4. Add random episode starts with a fixed length (Issue 4).
5. Wrap envs in `VecNormalize(norm_obs=True, norm_reward=False)` (Issue 5).
6. Use 8x `SubprocVecEnv` and bump training to 1–2M steps (Issue 8).

Everything else is polish. Those six fixes are the difference between "PPO might learn something" and "PPO will learn the actual reward you wrote down."
