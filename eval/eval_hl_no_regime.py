"""
eval/eval_hl_no_regime.py

Evaluates the no-regime HL stack (HL on top of no-regime fine-tuned LL)
with portfolio metrics. Compares against the full project leaderboard.
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

# CRITICAL: import from no_regime env
from env.portfolio_hrl_env_no_regime import (
    CoreConfig,
    HighLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)
from portfolio_stats import compute_stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LL_MODEL = (PATHS.checkpoints / "ll_finetune_real_no_regime"
            / "ll_ft_no_regime_final.zip")
LL_VECNORM = (PATHS.checkpoints / "ll_finetune_real_no_regime"
              / "ll_ft_no_regime_final_vecnorm.pkl")

PRETRAIN_DIR = PATHS.checkpoints / "hl_synth_pretrain_no_regime"
FT_DIR = PATHS.checkpoints / "hl_finetune_real_no_regime"

# (label, model_path, vecnorm_path)
HL_CHECKPOINTS = [
    # Synth pretrain endpoints
    (
        "hl_no_regime_pretrain_best_test",
        PRETRAIN_DIR / "best_on_real_test" / "best_model.zip",
        PRETRAIN_DIR / "ppo_hl_no_regime_vecnormalize_200000_steps.pkl",
    ),
    (
        "hl_no_regime_pretrain_700k_best_test_late",
        PRETRAIN_DIR / "ppo_hl_no_regime_700000_steps.zip",
        PRETRAIN_DIR / "ppo_hl_no_regime_vecnormalize_700000_steps.pkl",
    ),
    (
        "hl_no_regime_pretrain_final",
        PRETRAIN_DIR / "hl_no_regime_final.zip",
        PRETRAIN_DIR / "hl_no_regime_final_vecnorm.pkl",
    ),
    # Fine-tune endpoints
    (
        "hl_no_regime_ft_best_test",
        FT_DIR / "best_on_real_test" / "best_model.zip",
        FT_DIR / "ppo_hl_ft_no_regime_vecnormalize_200000_steps.pkl",
    ),
    (
        "hl_no_regime_ft_best_train",
        FT_DIR / "best_on_real_train" / "best_model.zip",
        FT_DIR / "ppo_hl_ft_no_regime_vecnormalize_200000_steps.pkl",
    ),
    (
        "hl_no_regime_ft_final",
        FT_DIR / "hl_ft_no_regime_final.zip",
        FT_DIR / "hl_ft_no_regime_final_vecnorm.pkl",
    ),
]

HL_CHECKPOINTS.insert(1, (
    "hl_no_regime_pretrain_100k",
    PRETRAIN_DIR / "ppo_hl_no_regime_100000_steps.zip",
    PRETRAIN_DIR / "ppo_hl_no_regime_vecnormalize_100000_steps.pkl",
))


N_SEEDS = 10
REAL_EPISODE_LENGTH = 384
SYNTH_EPISODE_LENGTH = 383
INITIAL_EQUITY = 1.0


# ---------------------------------------------------------------------------
# Frozen LL adapter
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
# Reporting
# ---------------------------------------------------------------------------

def aggregate(per_seed):
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    print(f"  {'label @ dataset':<46s}" + "".join(f"  {c:>7s}" for c in cols))
    print("  " + "-" * 46 + "".join(f"  {'-'*7}" for _ in cols))


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
    s = f"  {label:<46s}"
    for key, w, d, sign in cols:
        s += f"  {m[key]:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print("HL NO-REGIME evaluation — full HRL stack with no regime probs")
    print("=" * 110)
    print()
    print("This evaluates the final piece of the no-regime experiment:")
    print("  - LL was retrained without regime probs (309 features instead of 313)")
    print("  - HL pretrained on top of fine-tuned no-regime LL")
    print("  - HL fine-tuned on real_train")
    print()
    print("Best LL alone (no-regime): eq=1.524, Sharpe=+1.13, max_dd=0.314")
    print("Question: does the HL stack improve on this LL alone?")
    print()

    # Verify
    print("Verifying checkpoint files...")
    valid = []
    for label, model_p, vecnorm_p in HL_CHECKPOINTS:
        m_ok = model_p.exists()
        v_ok = vecnorm_p.exists()
        flag = "OK " if (m_ok and v_ok) else "MISSING"
        print(f"  [{flag}] {label}")
        if not m_ok: print(f"          model:    {model_p}")
        if not v_ok: print(f"          vecnorm:  {vecnorm_p}")
        if m_ok and v_ok:
            valid.append((label, model_p, vecnorm_p))

    if not LL_MODEL.exists() or not LL_VECNORM.exists():
        print(f"\n[MISSING] frozen LL")
        sys.exit(1)
    if not valid:
        print("\nNo valid checkpoints. Aborting.")
        sys.exit(1)

    # Datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")
    print(f"  synth: {pool['features'].shape}")

    # Frozen LL
    print("Loading frozen LL (no-regime fine-tuned)...")
    ll_predict_fn = load_frozen_ll()
    ll_adapter = FrozenLLAdapter(ll_predict_fn)

    # Eval each checkpoint
    all_results = {}
    for ckpt_label, model_path, vecnorm_path in valid:
        print()
        print("=" * 110)
        print(f"HL CHECKPOINT: {ckpt_label}")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 110)

        hl_model = PPO.load(str(model_path), device="cpu")
        ds_results = {}
        for ds_name, factory_builder in [
            ("real_train",
             lambda s: make_real_hl_env_factory(feats_train, rets_train, ll_adapter, s)),
            ("real_test",
             lambda s: make_real_hl_env_factory(feats_test, rets_test, ll_adapter, s)),
            ("synth",
             lambda s: make_synth_hl_env_factory(pool, ll_adapter, s + 9000)),
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

    # Headline
    print()
    print("=" * 110)
    print("HEADLINE: real_test — no-regime HL vs full leaderboard")
    print("=" * 110)
    print()
    print("Reference numbers (from previous evals):")
    print(f"  {'random':<42s}  eq=1.124  alpha=-0.577  sharpe=+0.45  max_dd=0.232")
    print(f"  {'synth_600k_LL (with regime probs)':<42s}  eq=1.434  alpha=-0.263  sharpe=+1.61  max_dd=0.110")
    print(f"  {'light_100k_LL (with regime probs, BEST)':<42s}  eq=1.531  alpha=-0.172  sharpe=+1.87  max_dd=0.112")
    print(f"  {'no-regime LL alone (best, ft_best_train)':<42s}  eq=1.524  alpha=-0.200  sharpe=+1.13  max_dd=0.314")
    print(f"  {'hl_v2_250k_best_test (with regime probs)':<42s}  eq=1.095  alpha=-0.600  sharpe=+0.46  max_dd=0.103")
    print(f"  {'hl_ft_uc_best_train (with regime, BEST HRL)':<42s}  eq=1.165  alpha=-0.521  sharpe=+0.51  max_dd=0.240")
    print()
    print_header()
    for ckpt_label, ds_results in all_results.items():
        print_row(f"{ckpt_label} @ real_test", ds_results["real_test"])

    # Train-test gap
    print()
    print("=" * 110)
    print("TRAIN-TEST GAP ANALYSIS")
    print("=" * 110)
    print()
    print(f"  {'checkpoint':<46s}  {'train_eq':>10s}  {'test_eq':>10s}  {'gap':>10s}")
    print(f"  {'-'*46}  {'-'*10}  {'-'*10}  {'-'*10}")
    for ckpt_label, ds_results in all_results.items():
        train_eq = ds_results["real_train"]["final_equity"]
        test_eq = ds_results["real_test"]["final_equity"]
        gap = train_eq - test_eq
        print(f"  {ckpt_label:<46s}  {train_eq:>10.3f}  {test_eq:>10.3f}  {gap:>+10.3f}")

    print()
    print("=" * 110)
    print("Done.")
    print("=" * 110)


if __name__ == "__main__":
    main()