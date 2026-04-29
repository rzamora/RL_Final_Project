"""
eval/eval_hl_pretrain.py

Evaluates the frozen-LL HL pretrain checkpoint on real test.

Reports the same portfolio metrics as eval_finetune.py for direct comparison
against the LL-only checkpoints. The HL was trained for 1M steps but did NOT
learn regime-conditional posture (see report). This eval gives concrete
final-equity and Sharpe numbers to include in the writeup as evidence of
why Fix B (LL retrain with randomized HL actions) was needed.

Run from project root:
    python eval/eval_hl_pretrain.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from project_config import PATHS
from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    HighLevelPortfolioEnv,
    LowLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)
from portfolio_stats import compute_stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HL_DIR = PATHS.checkpoints / "hl_synth_pretrain"
LL_DIR = PATHS.checkpoints / "finetune_light"

# Frozen LL (light_100k) — same as during HL training
LL_MODEL = LL_DIR / "best_on_real_test" / "best_model.zip"
LL_VECNORM = LL_DIR / "ppo_finetune_light_vecnormalize_100000_steps.pkl"

# HL checkpoints to compare. The 1M final + best-on-real-test (which by the
# log was step 300000 with eval_reward -12.99) + best-on-real-train.
HL_CHECKPOINTS = [
    (
        "hl_300k_best_test",
        HL_DIR / "best_on_real_test" / "best_model.zip",
        # Best-on-real-test was around step 300000 based on training log
        HL_DIR / "ppo_hl_vecnormalize_300000_steps.pkl",
    ),
    (
        "hl_1M_final",
        HL_DIR / "hl_pretrain_final.zip",
        HL_DIR / "hl_pretrain_final_vecnorm.pkl",
    ),
]

N_SEEDS = 10
REAL_EPISODE_LENGTH = 384
SYNTH_EPISODE_LENGTH = 383
INITIAL_EQUITY = 1.0


# ---------------------------------------------------------------------------
# Frozen LL adapter — same shape as in hl_synth_pretrain.py
# ---------------------------------------------------------------------------

import gymnasium as gym


class _DummyObsEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, obs_dim):
        super().__init__()
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=-1e6, high=1e6,
                                             shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                         shape=(4,), dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, a):
        return (np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0, True, False, {})


def load_frozen_ll():
    ll_model = PPO.load(str(LL_MODEL), device="cpu")
    dummy_env = DummyVecEnv([lambda: _DummyObsEnv(ll_model.observation_space.shape[0])])
    ll_vecnorm = VecNormalize.load(str(LL_VECNORM), dummy_env)
    ll_vecnorm.training = False
    # Same Fix A patch as during HL training
    ll_vecnorm.obs_rms.mean[-2:] = 0.0
    ll_vecnorm.obs_rms.var[-2:] = 1.0

    def predict_fn(obs):
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        normalized = ll_vecnorm.normalize_obs(obs_batch)
        action, _ = ll_model.predict(normalized, deterministic=True)
        return action[0]
    return predict_fn


class FrozenLLAdapter:
    def __init__(self, predict_fn):
        self._predict_fn = predict_fn
    def predict(self, obs, deterministic=True):
        return self._predict_fn(obs), None


# ---------------------------------------------------------------------------
# HL episode runner
# ---------------------------------------------------------------------------

def run_hl_episode(hl_model, hl_vecnorm_path, env_factory):
    """Run one HL episode through env_factory. Returns metrics dict."""
    vec = DummyVecEnv([env_factory])
    vec = VecNormalize.load(str(hl_vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False

    obs = vec.reset()
    equity = [INITIAL_EQUITY]
    bench = [INITIAL_EQUITY]
    turnover = []
    total_reward = 0.0
    done = False
    while not done:
        action, _ = hl_model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec.step(action)
        info = infos[0]
        equity.append(info["equity"])
        bench.append(info["bench_equity"])
        turnover.append(info["turnover"])
        total_reward += float(reward[0])
        done = bool(dones[0])
    vec.close()

    equity = np.array(equity)
    bench = np.array(bench)
    turnover = np.array(turnover)
    stats = compute_stats(equity, label="hl_agent", turnover=turnover)
    stats_bench = compute_stats(bench, label="bench")
    return {
        "final_equity": float(equity[-1]),
        "bench_final": float(bench[-1]),
        "alpha": float(equity[-1] - bench[-1]),
        "sharpe": stats.sharpe,
        "sortino": stats.sortino,
        "calmar": stats.calmar,
        "max_dd": stats.max_drawdown,
        "hit_rate": stats.hit_rate,
        "total_reward": total_reward,
    }


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def make_real_hl_env_factory(features, returns, ll_adapter, seed):
    def _init():
        cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)
        core = PortfolioCore(features, returns, cfg=cfg,
                              rng=np.random.default_rng(seed))
        return HighLevelPortfolioEnv(core, ll_adapter)
    return _init


def make_synth_hl_env_factory(pool, ll_adapter, seed):
    def _init():
        cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)
        sampler = SyntheticPoolCoreSampler(pool=pool, cfg=cfg,
                                            rng=np.random.default_rng(seed))
        return HighLevelPortfolioEnv(sampler, ll_adapter)
    return _init


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def aggregate(per_seed):
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    print(f"  {'label @ dataset':<28s}" + "".join(f"  {c:>7s}" for c in cols))
    print("  " + "-" * 28 + "".join(f"  {'-'*7}" for _ in cols))


def print_row(label, m):
    cols = [
        ("final_equity", 7, 3, ""),
        ("alpha",        7, 3, "+"),
        ("sharpe",       7, 2, "+"),
        ("sortino",      7, 2, "+"),
        ("calmar",       7, 2, "+"),
        ("max_dd",       7, 3, ""),
        ("hit_rate",     7, 3, ""),
        ("total_reward", 7, 1, "+"),
    ]
    s = f"  {label:<28s}"
    for key, w, d, sign in cols:
        s += f"  {m[key]:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("HL pretrain evaluation (frozen LL = light_100k)")
    print("=" * 80)

    # Verify files
    print("\nVerifying checkpoint files...")
    for label, model_p, vecnorm_p in HL_CHECKPOINTS:
        m_ok = model_p.exists()
        v_ok = vecnorm_p.exists()
        flag = "OK " if (m_ok and v_ok) else "MISSING"
        print(f"  [{flag}] {label}")
        if not m_ok:
            print(f"          model:    {model_p}")
        if not v_ok:
            print(f"          vecnorm:  {vecnorm_p}")
    if not LL_MODEL.exists() or not LL_VECNORM.exists():
        print(f"  [MISSING] frozen LL")
        sys.exit(1)

    # Filter to existing checkpoints
    valid_ckpts = [c for c in HL_CHECKPOINTS if c[1].exists() and c[2].exists()]
    if not valid_ckpts:
        print("\nNo valid HL checkpoints found. Aborting.")
        sys.exit(1)

    # Load datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)

    # Frozen LL
    print("\nLoading frozen LL (light_100k) with Fix A patch...")
    ll_predict_fn = load_frozen_ll()
    ll_adapter = FrozenLLAdapter(ll_predict_fn)

    # Eval each HL checkpoint on each dataset
    all_results = {}
    for ckpt_label, model_path, vecnorm_path in valid_ckpts:
        print()
        print("=" * 80)
        print(f"HL CHECKPOINT: {ckpt_label}")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 80)

        hl_model = PPO.load(str(model_path), device="cpu")

        ds_results = {}
        for ds_name, factory_builder in [
            ("real_train", lambda s: make_real_hl_env_factory(feats_train, rets_train, ll_adapter, s)),
            ("real_test",  lambda s: make_real_hl_env_factory(feats_test, rets_test, ll_adapter, s)),
            ("synth",      lambda s: make_synth_hl_env_factory(pool, ll_adapter, s + 9000)),
        ]:
            per_seed = []
            for seed in range(N_SEEDS):
                m = run_hl_episode(hl_model, vecnorm_path, factory_builder(seed))
                per_seed.append(m)
            ds_results[ds_name] = aggregate(per_seed)

        print_header()
        for ds in ["real_train", "real_test", "synth"]:
            print_row(f"{ckpt_label} @ {ds}", ds_results[ds])

        all_results[ckpt_label] = ds_results

    # Headline real_test
    print()
    print("=" * 80)
    print("HEADLINE: real_test (frozen-LL HL vs LL-only baselines)")
    print("=" * 80)
    print()
    print("Reference numbers from eval_finetune.py for context:")
    print("  random          @ real_test   eq=1.124  alpha=-0.577  sharpe=+0.45  max_dd=0.232")
    print("  synth_600k_LL   @ real_test   eq=1.434  alpha=-0.263  sharpe=+1.61  max_dd=0.110")
    print("  light_100k_LL   @ real_test   eq=1.531  alpha=-0.172  sharpe=+1.87  max_dd=0.112")
    print()
    print_header()
    for ckpt_label, ds_results in all_results.items():
        print_row(f"{ckpt_label} @ real_test", ds_results["real_test"])

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()