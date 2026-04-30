"""
eval/eval_hl_V2_finetuneHL.py

Evaluates the HL real-data fine-tune checkpoints with portfolio metrics.
Two fine-tune runs are evaluated:
  - hl_finetune_real_unconstrained (started from hl_v2_final, gross unconstrained)
  - hl_finetune_real_constrained_gross (started from hl_v2_cg_final, gross [0.8, 1.0])

For each run, three checkpoints are evaluated:
  - best_on_real_test (likely 100k by reward — least overfit point)
  - best_on_real_train (likely 200k by reward — most overfit point)
  - final 200k

Reference baselines printed alongside for the writeup.
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

# IMPORTANT: import the env class for each variant from the right module.
# Unconstrained uses portfolio_hrl_env_fixed; constrained-gross uses
# portfolio_hrl_env_constrained_gross. They have different parse_hl_action
# implementations, so loading the wrong env class would give wrong results.
from env.portfolio_hrl_env_fixed import (
    CoreConfig as CoreConfigUC,
    HighLevelPortfolioEnv as HLEnvUC,
    PortfolioCore as PortfolioCoreUC,
    SyntheticPoolCoreSampler as SamplerUC,
    load_synthetic_pool,
    process_raw_df,
)
from env.portfolio_hrl_env_constrained_gross import (
    CoreConfig as CoreConfigCG,
    HighLevelPortfolioEnv as HLEnvCG,
    PortfolioCore as PortfolioCoreCG,
    SyntheticPoolCoreSampler as SamplerCG,
)

from portfolio_stats import compute_stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LL_MODEL = (PATHS.checkpoints / "ll_random_hl_finetune"
            / "ll_random_hl_ft_final.zip")
LL_VECNORM = (PATHS.checkpoints / "ll_random_hl_finetune"
              / "ll_random_hl_ft_final_vecnorm.pkl")

UC_DIR = PATHS.checkpoints / "hl_finetune_real_unconstrained"
CG_DIR = PATHS.checkpoints / "hl_finetune_real_constrained_gross"

# (label, model_path, vecnorm_path, env_variant)
# env_variant in {"uc", "cg"} selects which env classes to use
HL_FT_CHECKPOINTS = [
    # Unconstrained fine-tune
    (
        "hl_ft_uc_best_test",
        UC_DIR / "best_on_real_test" / "best_model.zip",
        UC_DIR / "ppo_hl_ft_uc_vecnormalize_100000_steps.pkl",
        "uc",
    ),
    (
        "hl_ft_uc_best_train",
        UC_DIR / "best_on_real_train" / "best_model.zip",
        UC_DIR / "ppo_hl_ft_uc_vecnormalize_200000_steps.pkl",
        "uc",
    ),
    (
        "hl_ft_uc_final",
        UC_DIR / "hl_ft_uc_final.zip",
        UC_DIR / "hl_ft_uc_final_vecnorm.pkl",
        "uc",
    ),
    # Constrained-gross fine-tune
    (
        "hl_ft_cg_best_test",
        CG_DIR / "best_on_real_test" / "best_model.zip",
        CG_DIR / "ppo_hl_ft_cg_vecnormalize_100000_steps.pkl",
        "cg",
    ),
    (
        "hl_ft_cg_best_train",
        CG_DIR / "best_on_real_train" / "best_model.zip",
        CG_DIR / "ppo_hl_ft_cg_vecnormalize_200000_steps.pkl",
        "cg",
    ),
    (
        "hl_ft_cg_final",
        CG_DIR / "hl_ft_cg_final.zip",
        CG_DIR / "hl_ft_cg_final_vecnorm.pkl",
        "cg",
    ),
]

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
# HL episode runner
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
# Env factories — variant-aware (unconstrained vs constrained-gross)
# ---------------------------------------------------------------------------

def make_real_hl_env_factory(features, returns, ll_adapter, seed, variant):
    if variant == "uc":
        cfg_cls = CoreConfigUC
        core_cls = PortfolioCoreUC
        env_cls = HLEnvUC
    elif variant == "cg":
        cfg_cls = CoreConfigCG
        core_cls = PortfolioCoreCG
        env_cls = HLEnvCG
    else:
        raise ValueError(f"Unknown variant: {variant}")

    def _init():
        cfg = cfg_cls(episode_length=REAL_EPISODE_LENGTH)
        core = core_cls(features, returns, cfg=cfg,
                         rng=np.random.default_rng(seed))
        return env_cls(core, ll_adapter)
    return _init


def make_synth_hl_env_factory(pool, ll_adapter, seed, variant):
    if variant == "uc":
        cfg_cls = CoreConfigUC
        sampler_cls = SamplerUC
        env_cls = HLEnvUC
    elif variant == "cg":
        cfg_cls = CoreConfigCG
        sampler_cls = SamplerCG
        env_cls = HLEnvCG
    else:
        raise ValueError(f"Unknown variant: {variant}")

    def _init():
        cfg = cfg_cls(episode_length=SYNTH_EPISODE_LENGTH)
        sampler = sampler_cls(pool=pool, cfg=cfg,
                                rng=np.random.default_rng(seed))
        return env_cls(sampler, ll_adapter)
    return _init


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def aggregate(per_seed):
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    print(f"  {'label @ dataset':<36s}" + "".join(f"  {c:>7s}" for c in cols))
    print("  " + "-" * 36 + "".join(f"  {'-'*7}" for _ in cols))


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
    s = f"  {label:<36s}"
    for key, w, d, sign in cols:
        s += f"  {m[key]:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 96)
    print("HL real-data fine-tune evaluation")
    print("=" * 96)
    print()
    print("Two fine-tune runs:")
    print("  uc = unconstrained gross  (started from hl_v2_final.zip)")
    print("  cg = constrained gross    (started from hl_v2_cg_final.zip)")
    print()
    print("Both fine-tuned 200k steps on real_train data with:")
    print("  lr=3e-5, clip_range=0.05, ent_coef=0.01, n_epochs=4")
    print()

    # Verify
    print("Verifying checkpoint files...")
    valid = []
    for label, model_p, vecnorm_p, variant in HL_FT_CHECKPOINTS:
        m_ok = model_p.exists()
        v_ok = vecnorm_p.exists()
        flag = "OK " if (m_ok and v_ok) else "MISSING"
        print(f"  [{flag}] {label}")
        if not m_ok: print(f"          model:    {model_p}")
        if not v_ok: print(f"          vecnorm:  {vecnorm_p}")
        if m_ok and v_ok:
            valid.append((label, model_p, vecnorm_p, variant))

    if not LL_MODEL.exists() or not LL_VECNORM.exists():
        print(f"  [MISSING] frozen LL")
        sys.exit(1)
    if not valid:
        print("\nNo valid HL fine-tune checkpoints. Aborting.")
        sys.exit(1)

    # Datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)

    # Frozen LL
    print("Loading frozen LL (random-HL fine-tuned)...")
    ll_predict_fn = load_frozen_ll()
    ll_adapter = FrozenLLAdapter(ll_predict_fn)

    # Eval each checkpoint
    all_results = {}
    for ckpt_label, model_path, vecnorm_path, variant in valid:
        print()
        print("=" * 96)
        print(f"HL CHECKPOINT: {ckpt_label}  (variant={variant})")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 96)

        hl_model = PPO.load(str(model_path), device="cpu")
        ds_results = {}
        for ds_name, factory_builder in [
            ("real_train",
             lambda s: make_real_hl_env_factory(feats_train, rets_train, ll_adapter, s, variant)),
            ("real_test",
             lambda s: make_real_hl_env_factory(feats_test, rets_test, ll_adapter, s, variant)),
            ("synth",
             lambda s: make_synth_hl_env_factory(pool, ll_adapter, s + 9000, variant)),
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
    print("=" * 96)
    print("HEADLINE: real_test — HL fine-tunes vs full leaderboard")
    print("=" * 96)
    print()
    print("Reference numbers (from previous evals):")
    print(f"  {'random':<32s}  eq=1.124  alpha=-0.577  sharpe=+0.45  max_dd=0.232")
    print(f"  {'synth_600k_LL':<32s}  eq=1.434  alpha=-0.263  sharpe=+1.61  max_dd=0.110")
    print(f"  {'light_100k_LL (best baseline)':<32s}  eq=1.531  alpha=-0.172  sharpe=+1.87  max_dd=0.112")
    print(f"  {'hl_v1_300k_best_test':<32s}  eq=1.223  alpha=-0.478  sharpe=+1.60  max_dd=0.053")
    print(f"  {'hl_v2_250k_best_test (HL=0.03)':<32s}  eq=1.095  alpha=-0.600  sharpe=+0.46  max_dd=0.103")
    print(f"  {'hl_v2_cg_250k_best_test':<32s}  eq=1.084  alpha=-0.612  sharpe=+0.32  max_dd=0.338")
    print()
    print_header()
    for ckpt_label, ds_results in all_results.items():
        print_row(f"{ckpt_label} @ real_test", ds_results["real_test"])

    # Train-test gap analysis
    print()
    print("=" * 96)
    print("TRAIN-TEST GAP ANALYSIS")
    print("=" * 96)
    print()
    print(f"  {'checkpoint':<32s}  {'train_eq':>10s}  {'test_eq':>10s}  {'gap':>10s}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*10}")
    for ckpt_label, ds_results in all_results.items():
        train_eq = ds_results["real_train"]["final_equity"]
        test_eq = ds_results["real_test"]["final_equity"]
        gap = train_eq - test_eq
        print(f"  {ckpt_label:<32s}  {train_eq:>10.3f}  {test_eq:>10.3f}  {gap:>+10.3f}")

    print()
    print("=" * 96)
    print("Done.")
    print("=" * 96)


if __name__ == "__main__":
    main()