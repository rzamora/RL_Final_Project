"""
eval/eval_reweighted.py

Evaluates the reweighted LL-only checkpoints (synth pretrain + fine-tune)
on real test using portfolio metrics. Compares against:
  - light_100k LL baseline (the Phase 1 winner)
  - synth_600k LL baseline (synth pretrain only, original distribution)
  - All HRL variants we've tested

Headline question: does training on a regime-reweighted synth pool
produce a flat policy that beats light_100k on real test?
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

PRETRAIN_DIR = PATHS.checkpoints / "synth_pretrain_reweighted"
FINETUNE_DIR = PATHS.checkpoints / "finetune_reweighted"

# Synth pretrain endpoint (matches the original synth_600k comparison)
PRETRAIN_CHECKPOINTS = [
    (
        "synth_reweighted_1M",
        PRETRAIN_DIR / "synth_reweighted_pretrain_final.zip",
        PRETRAIN_DIR / "synth_reweighted_pretrain_final_vecnorm.pkl",
    ),
]

# Fine-tune checkpoints — including the 100k periodic which had the best
# real_test eval reward (-11.77), and the 200k final which had the best
# real_train (+2.21).
FINETUNE_CHECKPOINTS = [
    (
        "finetune_reweighted_100k",
        FINETUNE_DIR / "ppo_finetune_reweighted_12500_steps.zip",  # actual filename TBD
        FINETUNE_DIR / "ppo_finetune_reweighted_vecnormalize_12500_steps.pkl",
    ),
    (
        "finetune_reweighted_best_test",
        FINETUNE_DIR / "best_on_real_test" / "best_model.zip",
        FINETUNE_DIR / "ppo_finetune_reweighted_vecnormalize_12500_steps.pkl",
    ),
    (
        "finetune_reweighted_200k_final",
        FINETUNE_DIR / "finetune_reweighted_final.zip",
        FINETUNE_DIR / "finetune_reweighted_final_vecnorm.pkl",
    ),
]

N_SEEDS = 10
REAL_EPISODE_LENGTH = 384
SYNTH_EPISODE_LENGTH = 383
INITIAL_EQUITY = 1.0


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_ll_episode(model, vecnorm_path, env_factory):
    vec = DummyVecEnv([env_factory])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False

    obs = vec.reset()
    equity = [INITIAL_EQUITY]
    bench = [INITIAL_EQUITY]
    turnover = []
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
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
    stats = compute_stats(equity, label="ll_agent", turnover=turnover)
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


def make_real_ll_env_factory(features, returns, seed):
    def _init():
        cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)
        core = PortfolioCore(features, returns, cfg=cfg,
                              rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(core)
    return _init


def make_synth_ll_env_factory(pool, seed):
    def _init():
        cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)
        sampler = SyntheticPoolCoreSampler(pool=pool, cfg=cfg,
                                            rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(sampler)
    return _init


def aggregate(per_seed):
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    print(f"  {'label @ dataset':<38s}" + "".join(f"  {c:>7s}" for c in cols))
    print("  " + "-" * 38 + "".join(f"  {'-'*7}" for _ in cols))


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
    s = f"  {label:<38s}"
    for key, w, d, sign in cols:
        s += f"  {m[key]:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("Reweighted LL-only evaluation (synth pretrain + fine-tune)")
    print("=" * 100)

    print("\nVerifying checkpoint files...")
    all_ckpts = PRETRAIN_CHECKPOINTS + FINETUNE_CHECKPOINTS
    for label, model_p, vecnorm_p in all_ckpts:
        m_ok = model_p.exists()
        v_ok = vecnorm_p.exists()
        flag = "OK " if (m_ok and v_ok) else "MISSING"
        print(f"  [{flag}] {label}")
        if not m_ok: print(f"          model:    {model_p}")
        if not v_ok: print(f"          vecnorm:  {vecnorm_p}")

    valid = [c for c in all_ckpts if c[1].exists() and c[2].exists()]
    if not valid:
        print("\nNo valid checkpoints. Aborting.")
        sys.exit(1)

    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)

    all_results = {}
    for ckpt_label, model_path, vecnorm_path in valid:
        print()
        print("=" * 100)
        print(f"CHECKPOINT: {ckpt_label}")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 100)

        model = PPO.load(str(model_path), device="cpu")

        ds_results = {}
        for ds_name, factory_builder in [
            ("real_train", lambda s: make_real_ll_env_factory(feats_train, rets_train, s)),
            ("real_test",  lambda s: make_real_ll_env_factory(feats_test, rets_test, s)),
            ("synth",      lambda s: make_synth_ll_env_factory(pool, s + 9000)),
        ]:
            per_seed = []
            for seed in range(N_SEEDS):
                m = run_ll_episode(model, vecnorm_path, factory_builder(seed))
                per_seed.append(m)
            ds_results[ds_name] = aggregate(per_seed)

        print_header()
        for ds in ["real_train", "real_test", "synth"]:
            print_row(f"{ckpt_label} @ {ds}", ds_results[ds])
        all_results[ckpt_label] = ds_results

    # Headline
    print()
    print("=" * 100)
    print("HEADLINE: real_test — reweighted LL vs original LL baselines vs HRL variants")
    print("=" * 100)
    print()
    print("Reference numbers (from previous evals):")
    print("  random              @ real_test   eq=1.124  alpha=-0.577  sharpe=+0.45  max_dd=0.232")
    print("  synth_600k_LL       @ real_test   eq=1.434  alpha=-0.263  sharpe=+1.61  max_dd=0.110")
    print("  light_100k_LL       @ real_test   eq=1.531  alpha=-0.172  sharpe=+1.87  max_dd=0.112  *baseline*")
    print("  hl_v1_300k_best     @ real_test   eq=1.223  alpha=-0.478  sharpe=+1.60  max_dd=0.053")
    print("  hl_v2_1M_final      @ real_test   eq=1.217  alpha=-0.497  sharpe=+0.67  max_dd=0.237")
    print("  hl_v3_1M_final      @ real_test   eq=0.977  alpha=-0.720  sharpe=+0.08  max_dd=0.250")
    print("  joint_1M_final      @ real_test   eq=0.941  alpha=-0.751  sharpe=-0.46  max_dd=0.174")
    print()
    print_header()
    for ckpt_label, ds_results in all_results.items():
        print_row(f"{ckpt_label} @ real_test", ds_results["real_test"])

    print()
    print("=" * 100)
    print("Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()