"""
eval/eval_ll_no_regime.py

Evaluates the no-regime LL checkpoints with portfolio metrics and compares
against the original light_100k baseline (which used regime probs).

Same 10-seed methodology as eval_finetune.py.
Uses portfolio_hrl_env_no_regime env (309 features, no regime probs).
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

# CRITICAL: import from no_regime env, NOT from portfolio_hrl_env_fixed
from env.portfolio_hrl_env_no_regime import (
    CoreConfig,
    LowLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)
from portfolio_stats import compute_stats


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

PRETRAIN_DIR = PATHS.checkpoints / "ll_random_hl_synth_pretrain_no_regime"
FT_DIR = PATHS.checkpoints / "ll_finetune_real_no_regime"

CHECKPOINTS = [
    # No-regime synth pretrain endpoints
    (
        "ll_no_regime_pretrain_best_test",
        PRETRAIN_DIR / "best_on_real_test" / "best_model.zip",
        PRETRAIN_DIR / "ppo_ll_no_regime_vecnormalize_200000_steps.pkl",
    ),
    (
        "ll_no_regime_pretrain_best_train",
        PRETRAIN_DIR / "best_on_real_train" / "best_model.zip",
        PRETRAIN_DIR / "ppo_ll_no_regime_vecnormalize_400000_steps.pkl",
    ),
    (
        "ll_no_regime_pretrain_final",
        PRETRAIN_DIR / "ll_no_regime_pretrain_final.zip",
        PRETRAIN_DIR / "ll_no_regime_pretrain_final_vecnorm.pkl",
    ),
    # No-regime fine-tune endpoints
    (
        "ll_no_regime_ft_best_test",
        FT_DIR / "best_on_real_test" / "best_model.zip",
        FT_DIR / "ppo_ll_ft_no_regime_vecnormalize_50000_steps.pkl",
    ),
    (
        "ll_no_regime_ft_best_train",
        FT_DIR / "best_on_real_train" / "best_model.zip",
        FT_DIR / "ppo_ll_ft_no_regime_vecnormalize_100000_steps.pkl",
    ),
    (
        "ll_no_regime_ft_final",
        FT_DIR / "ll_ft_no_regime_final.zip",
        FT_DIR / "ll_ft_no_regime_final_vecnorm.pkl",
    ),
]

N_SEEDS = 10
REAL_EPISODE_LENGTH = 384
SYNTH_EPISODE_LENGTH = 383
INITIAL_EQUITY = 1.0


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode_with_curves(model, vec_env):
    obs = vec_env.reset()
    equity_list = [INITIAL_EQUITY]
    bench_list = [INITIAL_EQUITY]
    turnover_list = []
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        info = infos[0]
        equity_list.append(info["equity"])
        bench_list.append(info["bench_equity"])
        turnover_list.append(info["turnover"])
        total_reward += float(reward[0])
        done = bool(dones[0])
    return (np.array(equity_list), np.array(bench_list),
            np.array(turnover_list), total_reward)


def collect_metrics(equity, bench, turnover, total_reward, label="run"):
    stats_agent = compute_stats(equity, label=f"{label}_agent", turnover=turnover)
    return {
        "final_equity": float(equity[-1]),
        "bench_final": float(bench[-1]),
        "alpha": float(equity[-1] - bench[-1]),
        "sharpe": stats_agent.sharpe,
        "sortino": stats_agent.sortino,
        "calmar": stats_agent.calmar,
        "max_dd": stats_agent.max_drawdown,
        "hit_rate": stats_agent.hit_rate,
        "total_reward": total_reward,
    }


# ---------------------------------------------------------------------------
# Env builders — using no_regime env classes
# ---------------------------------------------------------------------------

def make_real_eval_vec(features, returns, seed, vecnorm_path):
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(seed))
        # Fixed HL action [0.33, 0.5] — same as eval_finetune.py protocol
        return LowLevelPortfolioEnv(core)

    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


def make_synth_eval_vec(pool, seed, vecnorm_path):
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def _init():
        sampler = SyntheticPoolCoreSampler(pool=pool, cfg=cfg,
                                              rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(sampler)

    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def aggregate(per_seed):
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    print(f"  {'label @ dataset':<40s}" + "".join(f"  {c:>7s}" for c in cols))
    print("  " + "-" * 40 + "".join(f"  {'-'*7}" for _ in cols))


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
    s = f"  {label:<40s}"
    for key, w, d, sign in cols:
        s += f"  {m[key]:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 105)
    print("LL NO-REGIME evaluation — portfolio metrics on no-regime checkpoints")
    print("=" * 105)
    print()
    print("Question: do the no-regime LL endpoints produce sensible portfolios")
    print("despite having very negative training rewards?")
    print()
    print("If yes (eq > 1.30 on real_test): reward looks bad but policy is fine.")
    print("If no (eq < 1.10 on real_test): training was pathological. Lower ent_coef")
    print("and retrain.")
    print()

    # ---------- Verify ----------
    print("Verifying checkpoint files...")
    valid = []
    for label, model_p, vecnorm_p in CHECKPOINTS:
        m_ok = model_p.exists()
        v_ok = vecnorm_p.exists()
        flag = "OK " if (m_ok and v_ok) else "MISSING"
        print(f"  [{flag}] {label}")
        if not m_ok: print(f"          model:    {model_p}")
        if not v_ok: print(f"          vecnorm:  {vecnorm_p}")
        if m_ok and v_ok:
            valid.append((label, model_p, vecnorm_p))

    if not valid:
        print("\nNo valid checkpoints. Aborting.")
        sys.exit(1)

    # ---------- Datasets ----------
    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")
    print(f"  synth: {pool['features'].shape}")

    # ---------- Each checkpoint ----------
    all_results = {}
    for ckpt_label, model_path, vecnorm_path in valid:
        print()
        print("=" * 105)
        print(f"CHECKPOINT: {ckpt_label}")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 105)

        model = PPO.load(str(model_path), device="cpu")
        ds_results = {}
        for ds_name, env_factory_vec in [
            ("real_train", lambda s: make_real_eval_vec(feats_train, rets_train, s, vecnorm_path)),
            ("real_test",  lambda s: make_real_eval_vec(feats_test, rets_test, s, vecnorm_path)),
            ("synth",      lambda s: make_synth_eval_vec(pool, s + 9000, vecnorm_path)),
        ]:
            per_seed = []
            for seed in range(N_SEEDS):
                vec = env_factory_vec(seed)
                try:
                    equity, bench, turnover, total_reward = run_episode_with_curves(model, vec)
                    per_seed.append(collect_metrics(equity, bench, turnover, total_reward,
                                                     label=f"{ckpt_label}_{ds_name}_{seed}"))
                finally:
                    vec.close()
            ds_results[ds_name] = aggregate(per_seed)

        print_header()
        for ds in ["real_train", "real_test", "synth"]:
            print_row(f"{ckpt_label} @ {ds}", ds_results[ds])
        all_results[ckpt_label] = ds_results

    # ---------- Headline ----------
    print()
    print("=" * 105)
    print("HEADLINE: real_test — no-regime LL vs original baselines")
    print("=" * 105)
    print()
    print("Original baselines (from previous evals):")
    print(f"  random                                    eq=1.124  alpha=-0.577  sharpe=+0.45  max_dd=0.232")
    print(f"  synth_600k_LL (with regime probs)         eq=1.434  alpha=-0.263  sharpe=+1.61  max_dd=0.110")
    print(f"  light_100k_LL (with regime probs, BEST)   eq=1.531  alpha=-0.172  sharpe=+1.87  max_dd=0.112")
    print()
    print_header()
    for ckpt_label, ds_results in all_results.items():
        print_row(f"{ckpt_label} @ real_test", ds_results["real_test"])

    print()
    print("=" * 105)
    print("Done.")
    print("=" * 105)


if __name__ == "__main__":
    main()