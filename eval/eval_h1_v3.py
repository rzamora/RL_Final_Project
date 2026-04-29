"""
eval/eval_hl_v3.py

Evaluates the HL v3 checkpoint (frozen LL = ll_regime_bucket_hl_finetune 100k)
on real test using portfolio metrics. Compares against:
  - HL v1 (frozen LL = light_100k, regime-blind LL underneath)
  - HL v2 (frozen LL = ll_random_hl_finetune, generalist LL underneath)
  - LL-only baselines (light_100k, synth_600k)
  - Random baseline

Headline question: does the regime-bucket-trained LL produce a foundation
that lets the HL beat the LL-alone baseline?
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from project_config import PATHS
from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    HighLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)
from portfolio_stats import compute_stats


# ---------------------------------------------------------------------------
# Paths — frozen LL = bucket-trained 100k
# ---------------------------------------------------------------------------

LL_MODEL = (PATHS.checkpoints / "ll_regime_bucket_hl_finetune"
            / "ppo_ll_regime_bucket_ft_100000_steps.zip")
LL_VECNORM = (PATHS.checkpoints / "ll_regime_bucket_hl_finetune"
              / "ppo_ll_regime_bucket_ft_vecnormalize_100000_steps.pkl")

HL_V3_DIR = PATHS.checkpoints / "hl_synth_pretrain_v3"

# Best on real_test: 750k (eval reward -4.26).
# Best on real_train: 750k (eval reward -4.26 also was new best on train).
# Final: 1M.
HL_V3_CHECKPOINTS = [
    (
        "hl_v3_750k_best_test",
        HL_V3_DIR / "best_on_real_test" / "best_model.zip",
        HL_V3_DIR / "ppo_hl_v3_vecnormalize_700000_steps.pkl",
    ),
    (
        "hl_v3_1M_final",
        HL_V3_DIR / "hl_v3_final.zip",
        HL_V3_DIR / "hl_v3_final_vecnorm.pkl",
    ),
]

N_SEEDS = 10
REAL_EPISODE_LENGTH = 384
SYNTH_EPISODE_LENGTH = 383
INITIAL_EQUITY = 1.0


# ---------------------------------------------------------------------------
# Frozen LL adapter (no Fix A patch — LL trained on full diagonal of [-1,+1]^2)
# ---------------------------------------------------------------------------

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
# Episode runner
# ---------------------------------------------------------------------------

def run_hl_episode(hl_model, hl_vecnorm_path, env_factory):
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
# Reporting
# ---------------------------------------------------------------------------

def aggregate(per_seed):
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    print(f"  {'label @ dataset':<32s}" + "".join(f"  {c:>7s}" for c in cols))
    print("  " + "-" * 32 + "".join(f"  {'-'*7}" for _ in cols))


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
    s = f"  {label:<32s}"
    for key, w, d, sign in cols:
        s += f"  {m[key]:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 88)
    print("HL v3 evaluation (frozen LL = ll_regime_bucket_hl_finetune 100k)")
    print("=" * 88)

    print("\nVerifying checkpoint files...")
    for label, model_p, vecnorm_p in HL_V3_CHECKPOINTS:
        m_ok = model_p.exists()
        v_ok = vecnorm_p.exists()
        flag = "OK " if (m_ok and v_ok) else "MISSING"
        print(f"  [{flag}] {label}")
        if not m_ok: print(f"          model:    {model_p}")
        if not v_ok: print(f"          vecnorm:  {vecnorm_p}")
    if not LL_MODEL.exists() or not LL_VECNORM.exists():
        print(f"  [MISSING] frozen LL")
        sys.exit(1)
    valid = [c for c in HL_V3_CHECKPOINTS if c[1].exists() and c[2].exists()]
    if not valid:
        print("\nNo valid HL v3 checkpoints. Aborting.")
        sys.exit(1)

    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)

    print("\nLoading frozen LL (regime-bucket fine-tuned 100k)...")
    ll_predict_fn = load_frozen_ll()
    ll_adapter = FrozenLLAdapter(ll_predict_fn)

    all_results = {}
    for ckpt_label, model_path, vecnorm_path in valid:
        print()
        print("=" * 88)
        print(f"HL CHECKPOINT: {ckpt_label}")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 88)

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

    print()
    print("=" * 88)
    print("HEADLINE: real_test — HL v3 vs HL v2 vs HL v1 vs LL-only vs random")
    print("=" * 88)
    print()
    print("Reference numbers (from previous evals):")
    print("  random              @ real_test   eq=1.124  alpha=-0.577  sharpe=+0.45  max_dd=0.232")
    print("  synth_600k_LL       @ real_test   eq=1.434  alpha=-0.263  sharpe=+1.61  max_dd=0.110")
    print("  light_100k_LL       @ real_test   eq=1.531  alpha=-0.172  sharpe=+1.87  max_dd=0.112")
    print("  hl_v1_300k_best_test@ real_test   eq=1.223  alpha=-0.478  sharpe=+1.60  max_dd=0.053")
    print("  hl_v1_1M_final      @ real_test   eq=1.059  alpha=-0.650  sharpe=+0.35  max_dd=0.097")
    print("  hl_v2_900k_best_test@ real_test   eq=1.143  alpha=-0.557  sharpe=+0.62  max_dd=0.085")
    print("  hl_v2_1M_final      @ real_test   eq=1.217  alpha=-0.497  sharpe=+0.67  max_dd=0.237")
    print()
    print_header()
    for ckpt_label, ds_results in all_results.items():
        print_row(f"{ckpt_label} @ real_test", ds_results["real_test"])

    print()
    print("=" * 88)
    print("Done.")
    print("=" * 88)


if __name__ == "__main__":
    main()