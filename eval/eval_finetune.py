"""
eval/eval_finetune.py

Compare synth-pretrain baseline (600k) against light fine-tune checkpoints.
Same 10-seed methodology as eval_synth_pretrain.py — uses portfolio_stats.compute_stats.

After heavy fine-tune is run, add its checkpoints to CHECKPOINTS list and rerun.

Run from project root:
    python eval/eval_finetune.py
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
# Checkpoints to compare
# ---------------------------------------------------------------------------

SYNTH_DIR = PATHS.checkpoints / "synth_pretrain"
LIGHT_DIR = PATHS.checkpoints / "finetune_light"
HEAVY_DIR = PATHS.checkpoints / "finetune_heavy"

CHECKPOINTS = [
    # Baseline: synth pretrain 600k
    (
        "synth_600k_baseline",
        SYNTH_DIR / "best_on_real_train" / "best_model.zip",
        SYNTH_DIR / "ppo_synth_vecnormalize_600000_steps.pkl",
    ),
    # Light fine-tune
    (
        "light_100k_best_test",
        LIGHT_DIR / "best_on_real_test" / "best_model.zip",
        LIGHT_DIR / "ppo_finetune_light_vecnormalize_100000_steps.pkl",
    ),
    (
        "light_200k_best_train",
        LIGHT_DIR / "best_on_real_train" / "best_model.zip",
        LIGHT_DIR / "ppo_finetune_light_vecnormalize_200000_steps.pkl",
    ),
    (
        "light_final",
        LIGHT_DIR / "finetune_light_final.zip",
        LIGHT_DIR / "finetune_light_final_vecnorm.pkl",
    ),
    # Heavy fine-tune
    (
        "heavy_200k_best_test",
        HEAVY_DIR / "best_on_real_test" / "best_model.zip",
        HEAVY_DIR / "ppo_finetune_heavy_vecnormalize_200000_steps.pkl",
    ),
    (
        "heavy_500k_best_train",
        HEAVY_DIR / "best_on_real_train" / "best_model.zip",
        HEAVY_DIR / "ppo_finetune_heavy_vecnormalize_500000_steps.pkl",
    ),
    (
        "heavy_final",
        HEAVY_DIR / "finetune_heavy_final.zip",
        HEAVY_DIR / "finetune_heavy_final_vecnorm.pkl",
    ),
]

N_SEEDS = 10
REAL_EPISODE_LENGTH = 384
SYNTH_EPISODE_LENGTH = 383
INITIAL_EQUITY = 1.0


# ---------------------------------------------------------------------------
# Episode runners (same as eval_synth_pretrain.py)
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


def run_episode_random(env):
    obs, _ = env.reset()
    equity_list = [INITIAL_EQUITY]
    bench_list = [INITIAL_EQUITY]
    turnover_list = []
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        equity_list.append(info["equity"])
        bench_list.append(info["bench_equity"])
        turnover_list.append(info["turnover"])
        total_reward += reward
        done = term or trunc
    return (np.array(equity_list), np.array(bench_list),
            np.array(turnover_list), total_reward)


def collect_metrics(equity, bench, turnover, total_reward, label="run"):
    stats_agent = compute_stats(equity, label=f"{label}_agent", turnover=turnover)
    stats_bench = compute_stats(bench, label=f"{label}_bench")
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
# Env builders
# ---------------------------------------------------------------------------

def make_real_eval_vec(features, returns, seed, vecnorm_path):
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(core)

    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


def make_synth_eval_vec(pool, seed, vecnorm_path):
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def _init():
        sampler = SyntheticPoolCoreSampler(pool, cfg=cfg, rng=np.random.default_rng(seed))
        return LowLevelPortfolioEnv(sampler)

    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


def make_real_random_env(features, returns, seed):
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)
    core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(seed))
    return LowLevelPortfolioEnv(core)


def make_synth_random_env(pool, seed):
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)
    sampler = SyntheticPoolCoreSampler(pool, cfg=cfg, rng=np.random.default_rng(seed))
    return LowLevelPortfolioEnv(sampler)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def aggregate(per_seed):
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    print(f"  {'label @ dataset':<30s}" + "".join(f"  {c:>7s}" for c in cols))
    print("  " + "-" * 30 + "".join(f"  {'-'*7}" for _ in cols))


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
    s = f"  {label:<30s}"
    for key, w, d, sign in cols:
        s += f"  {m[key]:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 95)
    print("Fine-tune evaluation: synth baseline vs light fine-tune checkpoints")
    print("=" * 95)

    # ---------- Verify files ----------
    print("\nVerifying checkpoint files...")
    missing = []
    for label, model_p, vecnorm_p in CHECKPOINTS:
        m_ok = model_p.exists()
        v_ok = vecnorm_p.exists()
        flag = "OK " if (m_ok and v_ok) else "MISSING"
        print(f"  [{flag}] {label}")
        if not m_ok:
            print(f"          model:    {model_p}")
            missing.append(label)
        if not v_ok:
            print(f"          vecnorm:  {vecnorm_p}")
            missing.append(label)
    if missing:
        print(f"\n{len(set(missing))} checkpoint(s) missing files. Aborting.")
        sys.exit(1)

    # ---------- Datasets ----------
    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)

    # ---------- Random baseline ----------
    print(f"\nComputing random baseline ({N_SEEDS} seeds × 3 datasets)...")
    rand_results = {}
    for ds_name, env_factory in [
        ("real_train", lambda s: make_real_random_env(feats_train, rets_train, s)),
        ("real_test",  lambda s: make_real_random_env(feats_test, rets_test, s)),
        ("synth",      lambda s: make_synth_random_env(pool, s + 9000)),
    ]:
        per_seed = []
        for seed in range(N_SEEDS):
            env = env_factory(seed)
            equity, bench, turnover, total_reward = run_episode_random(env)
            per_seed.append(collect_metrics(equity, bench, turnover, total_reward,
                                             label=f"rand_{ds_name}_{seed}"))
        rand_results[ds_name] = aggregate(per_seed)

    print()
    print("=" * 95)
    print("RANDOM BASELINE")
    print("=" * 95)
    print_header()
    for ds in ["real_train", "real_test", "synth"]:
        print_row(f"random @ {ds}", rand_results[ds])

    # ---------- Each checkpoint on each dataset ----------
    all_results = {}
    for ckpt_label, model_path, vecnorm_path in CHECKPOINTS:
        print()
        print("=" * 95)
        print(f"CHECKPOINT: {ckpt_label}")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 95)

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

    # ---------- Headline: real_test comparison ----------
    print()
    print("=" * 95)
    print("HEADLINE: real_test (the metric that matters)")
    print("=" * 95)
    print_header()
    print_row("random @ real_test", rand_results["real_test"])
    for ckpt_label, ds_results in all_results.items():
        print_row(f"{ckpt_label} @ real_test", ds_results["real_test"])

    # ---------- Deltas vs synth baseline on real_test ----------
    print()
    print("=" * 95)
    print("DELTAS vs synth_600k baseline on real_test")
    print("=" * 95)
    if "synth_600k_baseline" in all_results:
        baseline = all_results["synth_600k_baseline"]["real_test"]
        print(f"  baseline: eq={baseline['final_equity']:.3f}  "
              f"alpha={baseline['alpha']:+.3f}  "
              f"sharpe={baseline['sharpe']:+.2f}  "
              f"max_dd={baseline['max_dd']:.3f}")
        print()
        for ckpt_label, ds_results in all_results.items():
            if ckpt_label == "synth_600k_baseline":
                continue
            m = ds_results["real_test"]
            d_eq = m["final_equity"] - baseline["final_equity"]
            d_alpha = m["alpha"] - baseline["alpha"]
            d_sharpe = m["sharpe"] - baseline["sharpe"]
            d_dd = m["max_dd"] - baseline["max_dd"]
            print(f"  {ckpt_label:<25s}  "
                  f"Δeq={d_eq:+.3f}  "
                  f"Δalpha={d_alpha:+.3f}  "
                  f"Δsharpe={d_sharpe:+.2f}  "
                  f"Δmax_dd={d_dd:+.3f}")

    print()
    print("=" * 95)
    print("Done.")
    print("=" * 95)


if __name__ == "__main__":
    main()