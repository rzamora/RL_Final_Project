# Startup checklist — from zero to first training run

The plan: validate each layer in isolation before stacking the next. If something breaks, you know exactly where.

```
Step 0 — Environment setup           (15 min)
Step 1 — Data sanity                 (10 min)
Step 2 — Env smoke test              (15 min)
Step 3 — Random-policy baseline      (10 min)
Step 4 — Tiny LL training run        (30 min)
Step 5 — Reward signal check         (15 min)
Step 6 — Synthetic pool integration  (30 min)
Step 7 — Vectorized env scaling      (15 min)
Step 8 — First real training run     (overnight)
```

Total: ~3 hours of active work before the first overnight training run. **Do not skip steps.** Every one of them has caught a real bug in similar projects.

---

## Step 0 — Environment setup

```bash
# Python 3.11 recommended
pip install stable-baselines3[extra]==2.3.0
pip install gymnasium pandas numpy matplotlib
pip install torch  # match your CUDA version

# Sanity check the install
python -c "import stable_baselines3; print(stable_baselines3.__version__)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If CUDA isn't available, that's fine for steps 0–6 (CPU is plenty for tiny runs). You'll want GPU for step 8.

Project layout I'd suggest:

```
portfolio_rl/
├── env/
│   └── portfolio_hrl_env_fixed.py   # the fixed env from the review
├── eval/
│   └── portfolio_stats.py            # the benchmark module
├── data/
│   ├── train.csv                     # your real data
│   ├── test.csv
│   └── synth_pool.npz                # 2000-path synthetic pool
├── checkpoints/
└── tests/
    ├── test_01_data.py
    ├── test_02_env.py
    └── ...
```

---

## Step 1 — Data sanity

Goal: verify the data loads without surprises and matches what you expect.

```python
# tests/test_01_data.py
import pandas as pd
import numpy as np
from env.portfolio_hrl_env_fixed import process_raw_df, load_synthetic_pool

# Real data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print(f"Train: {len(train_df)} rows, {train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]}")
print(f"Test:  {len(test_df)} rows, {test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]}")

# Process
feats_train, rets_train, prices_train = process_raw_df(train_df)
feats_test, rets_test, prices_test = process_raw_df(test_df)

assert feats_train.shape[1] == 313, f"Expected 313 features, got {feats_train.shape[1]}"
assert feats_test.shape[1] == 313
assert rets_train.shape[1] == 4
print(f"Train features: {feats_train.shape}")
print(f"Test features:  {feats_test.shape}")

# Check for NaN/Inf
assert not np.any(np.isnan(feats_train)), "Train features have NaN"
assert not np.any(np.isinf(feats_train)), "Train features have Inf"
assert not np.any(np.isnan(feats_test))
print("No NaN/Inf in features")

# Check return ranges are sane
print(f"Train return ranges per asset:")
for i, name in enumerate(["NVDA", "AMD", "SMH", "TLT"]):
    r = rets_train[1:, i]  # skip row 0
    print(f"  {name}: min={r.min():.4f}, max={r.max():.4f}, std={r.std():.4f}")

# Synthetic pool
pool = load_synthetic_pool("data/synth_pool.npz")
print(f"\nPool features: {pool['features'].shape}")
print(f"Pool returns:  {pool['returns'].shape}")
assert pool["features"].shape[2] == 313, "Pool feature dim must match real (313)"
print("Pool feature dim matches real data ✓")
```

What to look for:
- Train ~4,579 rows, test ~830 rows
- 313 features in both
- No NaN/Inf
- Return std ~0.02–0.04 for equities, ~0.01 for TLT
- If pool features are 317 → loader didn't drop closes → fix `load_synthetic_pool` call

---

## Step 2 — Env smoke test

Goal: env constructs, runs an episode, doesn't crash, info dict is populated correctly.

```python
# tests/test_02_env.py
import numpy as np
from env.portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig,
    LowLevelPortfolioEnv,
    process_raw_df,
)
import pandas as pd

train_df = pd.read_csv("data/train.csv")
feats, rets, _ = process_raw_df(train_df)

cfg = CoreConfig(episode_length=384)
core = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(0))
env = LowLevelPortfolioEnv(core)

obs, info = env.reset(seed=0)
print(f"Obs shape: {obs.shape}")  # should be 313 + 4 + 4 + 1 + 2 = 324
print(f"Action space: {env.action_space}")
print(f"Episode start: t={core.t_start}, end: t={core.t_end}")

# Run one full episode with random actions
total_reward = 0.0
equity_curve = [core.equity]
for step in range(400):  # safety bound
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    equity_curve.append(info["equity"])
    if terminated or truncated:
        break

print(f"Episode finished after {step+1} steps")
print(f"Total reward: {total_reward:.4f}")
print(f"Final equity: {info['equity']:.4f}")
print(f"Final drawdown: {info['drawdown']:.4f}")

# Sanity checks
assert step + 1 == 384, f"Episode should run 384 steps, got {step+1}"
assert info["equity"] > 0, "Equity went non-positive — bug somewhere"
assert info["equity"] < 100, "Equity above 100 in 384 steps with random actions — also weird"
assert 0 <= info["drawdown"] <= 1, f"Drawdown out of [0,1]: {info['drawdown']}"
print("Smoke test passed ✓")
```

Common failures here:
- `obs.shape` not 324 → feature/state count mismatch → check `process_raw_df` output and `LowLevelPortfolioEnv` obs_dim calculation
- Episode runs more or fewer than 384 steps → off-by-one in `done` condition
- `equity` blows up to inf or goes negative → numerical issue in reward or dynamics
- `drawdown` > 1 → drawdown formula bug

---

## Step 3 — Random-policy baseline

Goal: get a reference number for "random allocation" performance. The trained agent should beat this. If it doesn't, training failed.

```python
# tests/test_03_random_baseline.py
import numpy as np
import pandas as pd
from env.portfolio_hrl_env_fixed import PortfolioCore, CoreConfig, LowLevelPortfolioEnv, process_raw_df
from eval.portfolio_stats import compute_stats, _format_single

train_df = pd.read_csv("data/train.csv")
feats, rets, _ = process_raw_df(train_df)

# Run 20 episodes with random actions, collect equity curves
equities = []
for seed in range(20):
    core = PortfolioCore(feats, rets, cfg=CoreConfig(episode_length=384),
                         rng=np.random.default_rng(seed))
    env = LowLevelPortfolioEnv(core)
    obs, _ = env.reset(seed=seed)

    eq = [core.equity]
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        eq.append(info["equity"])
        done = terminated or truncated
    equities.append(np.array(eq))

# Aggregate stats across 20 random episodes
final_equities = [e[-1] for e in equities]
print(f"Random policy, 20 episodes of 384 days each:")
print(f"  Final equity: mean={np.mean(final_equities):.3f}, std={np.std(final_equities):.3f}")
print(f"  Min final:    {min(final_equities):.3f}")
print(f"  Max final:    {max(final_equities):.3f}")

# Compute Sharpe for one of them
stats = compute_stats(equities[0], label="Random policy (1 episode)")
print(_format_single(stats))
```

What to expect: random policy on 384-day windows from train data should give final equity all over the map (0.4 to 3.0+ range), Sharpe near 0, often negative. If random policy reliably wins, your reward function is rewarding noise.

**Save these numbers.** They're your "did training do anything" benchmark.

---

## Step 4 — Tiny LL training run

Goal: PPO can actually learn something. Don't aim for great performance — aim for **reward going up over time**.

```python
# tests/test_04_tiny_train.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig, LowLevelPortfolioEnv,
    LayerNormActorCriticPolicy, process_raw_df,
)

train_df = pd.read_csv("data/train.csv")
feats, rets, _ = process_raw_df(train_df)
cfg = CoreConfig(episode_length=384)

def make_env(seed):
    def _init():
        core = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(core)
    return _init

vec = DummyVecEnv([make_env(i) for i in range(2)])
vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

model = PPO(
    LayerNormActorCriticPolicy,
    vec,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    verbose=1,
    seed=0,
)

# 50k steps — should take 5–15 min on CPU
model.learn(total_timesteps=50_000)
model.save("checkpoints/test_tiny_ll")
vec.save("checkpoints/test_tiny_ll_vecnorm.pkl")

print("Tiny training done.")
```

What to watch in the verbose output:
- `ep_rew_mean` — should trend **upward** over the run. If it's flat or trending down, there's a problem.
- `ep_len_mean` — should be exactly 384.
- `value_loss` and `policy_loss` — should be decreasing-ish but PPO is noisy. Don't panic at fluctuations.
- `clip_fraction` — should settle around 0.1–0.3. If it's pinned at 0 or 1, learning rate is too small/big.
- `entropy_loss` — should slowly become more negative (entropy decreasing, policy becoming more deterministic).

**Red flags:**
- `ep_rew_mean` decreasing → reward sign error, or env broken
- `ep_rew_mean = NaN` → numerical blowup, check observations and rewards for inf/NaN
- `value_loss` increasing dramatically → value function diverging, lower learning rate
- Training crashes with shape errors → feature dim mismatch between env and policy

---

## Step 5 — Reward signal check

Goal: verify the trained policy actually learned something rather than just exploiting a reward bug.

```python
# tests/test_05_reward_check.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import pandas as pd

from env.portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig, LowLevelPortfolioEnv, process_raw_df,
)
from eval.portfolio_stats import compute_stats, _format_single

train_df = pd.read_csv("data/train.csv")
feats, rets, _ = process_raw_df(train_df)
cfg = CoreConfig(episode_length=384)

# Reload the trained model
def make_env(seed):
    def _init():
        core = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(core)
    return _init

eval_vec = DummyVecEnv([make_env(99)])
eval_vec = VecNormalize.load("checkpoints/test_tiny_ll_vecnorm.pkl", eval_vec)
eval_vec.training = False
eval_vec.norm_reward = False

model = PPO.load("checkpoints/test_tiny_ll", env=eval_vec)

# Compare: trained policy vs random policy on the same 20 starting points
trained_equities = []
random_equities = []

for seed in range(20):
    # Trained
    core_t = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(seed))
    env_t = LowLevelPortfolioEnv(core_t)
    obs, _ = env_t.reset(seed=seed)
    eq_t = [core_t.equity]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env_t.step(action)
        eq_t.append(info["equity"])
        done = terminated or truncated
    trained_equities.append(np.array(eq_t))

    # Random — same start
    core_r = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(seed))
    env_r = LowLevelPortfolioEnv(core_r)
    obs, _ = env_r.reset(seed=seed)
    eq_r = [core_r.equity]
    done = False
    while not done:
        action = env_r.action_space.sample()
        obs, _, terminated, truncated, info = env_r.step(action)
        eq_r.append(info["equity"])
        done = terminated or truncated
    random_equities.append(np.array(eq_r))

trained_finals = [e[-1] for e in trained_equities]
random_finals = [e[-1] for e in random_equities]

print(f"Trained (50k steps): mean final equity {np.mean(trained_finals):.3f} ± {np.std(trained_finals):.3f}")
print(f"Random:              mean final equity {np.mean(random_finals):.3f} ± {np.std(random_finals):.3f}")
print(f"Wins (trained > random): {sum(t > r for t, r in zip(trained_finals, random_finals))}/20")
```

What to expect after only 50k steps: trained policy should win ~12–15 of 20 episodes. Not a landslide, but better than random's expected 10/20. If it's 10/20 or worse, something's off.

If the trained policy crushes random (19/20), don't celebrate yet — that's almost certainly because random's downside is dominated by the gross-leverage variance, and the trained policy just learned "go to low gross." That's a degenerate solution. To check, look at the trained policy's average gross exposure: if it's hugging 0, the policy is just sitting in cash. The reward needs to encourage actual investment.

---

## Step 6 — Synthetic pool integration

Goal: confirm training works on synthetic data the same way it does on real data, and the resulting policy can be loaded and run on real data without dimension errors.

```python
# tests/test_06_synth.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.portfolio_hrl_env_fixed import (
    SyntheticPoolCoreSampler, CoreConfig, LowLevelPortfolioEnv,
    LayerNormActorCriticPolicy, load_synthetic_pool,
    PortfolioCore, process_raw_df,
)
import pandas as pd

# Load synthetic pool
pool = load_synthetic_pool("data/synth_pool.npz", drop_close_features=True)
cfg = CoreConfig(episode_length=384)

# Train on synthetic — 50k steps
def make_synth_env(seed):
    def _init():
        sampler = SyntheticPoolCoreSampler(
            pool, cfg=cfg, rng=np.random.default_rng(seed)
        )
        return LowLevelPortfolioEnv(sampler)
    return _init

vec = DummyVecEnv([make_synth_env(i) for i in range(2)])
vec = VecNormalize(vec, norm_obs=True, norm_reward=False)

model = PPO(LayerNormActorCriticPolicy, vec, n_steps=512, verbose=1, seed=0)
model.learn(total_timesteps=50_000)

# Critical test: can the policy run on real data?
train_df = pd.read_csv("data/train.csv")
feats, rets, _ = process_raw_df(train_df)
core_real = PortfolioCore(feats, rets, cfg=cfg)
env_real = LowLevelPortfolioEnv(core_real)
obs, _ = env_real.reset(seed=0)

# This is the moment of truth — if feature dims don't match, predict() crashes
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, info = env_real.step(action)
    done = terminated or truncated

print(f"Synth-trained policy ran on real data successfully.")
print(f"Final equity on real test window: {info['equity']:.4f}")
```

If this runs without shape errors, you've confirmed that the synth-pretrain → real-finetune pipeline will work. If you get a shape mismatch, the most likely cause is that `load_synthetic_pool` didn't drop closes (giving 317-dim) while `process_raw_df` did (313-dim).

---

## Step 7 — Vectorized env scaling

Goal: confirm `SubprocVecEnv` works on your machine (some setups have issues with multiprocessing + pytorch).

```python
# tests/test_07_subproc.py
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from env.portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig, LowLevelPortfolioEnv,
    LayerNormActorCriticPolicy, process_raw_df,
)

train_df = pd.read_csv("data/train.csv")
feats, rets, _ = process_raw_df(train_df)
cfg = CoreConfig(episode_length=384)

def make_env(seed):
    def _init():
        core = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(core)
    return _init

# Test with 8 parallel envs
if __name__ == "__main__":
    vec = SubprocVecEnv([make_env(i) for i in range(8)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)

    model = PPO(LayerNormActorCriticPolicy, vec, n_steps=256, verbose=0, seed=0)

    t0 = time.time()
    model.learn(total_timesteps=20_000)
    elapsed = time.time() - t0
    print(f"20k steps with 8 SubprocVecEnv: {elapsed:.1f}s ({20_000/elapsed:.0f} steps/s)")
```

Compare to step 4's `DummyVecEnv` timing. With 8 SubprocVecEnvs you should see roughly 4–6x speedup over single env (not 8x — there's serialization overhead).

If `SubprocVecEnv` errors with pickling issues, fall back to `DummyVecEnv` for now and increase `n_steps` to compensate. Multiprocessing on Windows + Jupyter is particularly fragile.

**Note the `if __name__ == "__main__":` guard.** Required on Windows/macOS for SubprocVecEnv. Don't forget it.

---

## Step 8 — First real training run

Only do this after steps 1–7 pass. This is the actual experiment.

```python
# scripts/train_phase1_synth.py
from env.portfolio_hrl_env_fixed import train_pipeline

train_pipeline(
    train_csv_path="data/train.csv",
    synthetic_pool_path="data/synth_pool.npz",
    out_dir="checkpoints/seed0",
    seed=0,
    pretrain_steps=2_000_000,
    finetune_steps=500_000,
    n_envs_pretrain=8,
    n_envs_finetune=4,
)
```

Plan the run:
- ~4–8 hours on a single GPU + 8 vCPU machine
- ~12–16 hours on CPU only
- Run overnight, check in the morning

What to monitor (in TensorBoard or PPO's verbose output):
- `ep_rew_mean` should rise monotonically-ish through Phase 1, plateau, then **dip** when Phase 2 starts (real data is harder), then rise again.
- `ep_len_mean` always exactly 384.
- During fine-tune phase, `clip_fraction` may spike briefly because the new data distribution surprises the policy. Settles within ~50k steps.

When it finishes:

```python
# scripts/eval_phase3_test.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import pandas as pd

from env.portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig, HighLevelPortfolioEnv, process_raw_df,
)
from eval.portfolio_stats import (
    run_agent_rollout, run_equal_weight_benchmark, compare,
)

# Load test data
test_df = pd.read_csv("data/test.csv")
feats, rets, _ = process_raw_df(test_df)

# Load both models
ll_model = PPO.load("checkpoints/seed0/ll_real_finetune_seed0")
hl_model = PPO.load("checkpoints/seed0/hl_real_finetune_seed0")

# Make eval env — full test window, no random start, deterministic
cfg = CoreConfig(episode_length=len(feats) - 1)  # full test window
core = PortfolioCore(feats, rets, cfg=cfg)

# Wrap the LL model so HL env can use it
hl_env = HighLevelPortfolioEnv(core, ll_model=ll_model)

# Need to wrap in VecNormalize and load stats
hl_eval_vec = DummyVecEnv([lambda: hl_env])
hl_eval_vec = VecNormalize.load("checkpoints/seed0/hl_real_vecnorm_seed0.pkl", hl_eval_vec)
hl_eval_vec.training = False

# Roll out
agent_stats, agent_eq, weights, turnover = run_agent_rollout(hl_model, hl_env, deterministic=True)

# Benchmark
bench_stats, bench_eq = run_equal_weight_benchmark(rets[1:], label="EW(NVDA,AMD,SMH)")

# Compare
compare(agent_stats, bench_stats, agent_eq, bench_eq, dates=test_df["date"].values[1:])
```

This is your first out-of-sample number.

---

## What to do if results disappoint

The first training run will probably underperform the equal-weight benchmark on raw return. That's expected — the benchmark is brutal (Sharpe 1.70 on the test period). What you want to see:

| Outcome | Diagnosis |
|---|---|
| Agent matches benchmark return with lower drawdown | ✓ working as designed |
| Agent has 50–80% of benchmark return but Sharpe ≥ 1.70 | ✓ also working as designed |
| Agent loses money (negative total return) | reward function not learning |
| Agent has near-zero turnover, just sits in cash | turnover penalty too high or benchmark term dominating |
| Agent performance varies wildly across seeds (ranges by 10x) | overfitting, increase synth pretrain steps |
| Agent crashes during eval | feature dim or vecnorm stats mismatch |

For each failure mode, the diagnostic is to plot `weights` over the rollout. If weights are all near zero → cash-hoarding. If weights are all at gross 1.5 → leverage-pinning. If weights flip every day → no convergence. Each pattern points to a different fix.

---

## Tracking experiments

Make a CSV like this and update after every run:

| run_id | date | seed | pretrain_steps | finetune_steps | test_sharpe | test_max_dd | test_cagr | notes |
|---|---|---|---|---|---|---|---|---|
| 001 | 2026-04-29 | 0 | 2M | 500k | ? | ? | ? | first run |
| ... | | | | | | | | |

You'll want to look back at this in a month and remember why run 003 worked better than run 002. You won't remember without notes.

---

## What I'd actually do in the next 8 hours

1. **Now (next 30 min):** Steps 0 and 1. Get data loading clean.
2. **Next hour:** Steps 2 and 3. Env smoke test + random baseline numbers saved.
3. **Hour 2–3:** Step 4 (50k tiny train) + step 5 (verify it learned). This is the real test of the env.
4. **Hour 3–4:** Steps 6 and 7. Confirm synthetic pool integration and parallelism.
5. **Then:** kick off step 8 overnight. Sleep on it.
6. **Tomorrow morning:** look at numbers, iterate.

Don't go from step 1 directly to step 8. The intermediate steps will catch bugs that would otherwise waste 8 hours of GPU time.
