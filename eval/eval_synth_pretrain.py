"""
eval/eval_synth_pretrain.py

Evaluates synth-pretrained PPO checkpoints on three datasets:
  - Real train CSV (in-distribution for fine-tune)
  - Real test CSV (held-out, the number that actually matters)
  - Held-out synth paths (10 random samples from the pool, fresh seeds)

For each (checkpoint, dataset) pair runs 10 seeded episodes and reports
median final equity, alpha vs the env's internal EW benchmark, Sharpe,
Sortino, Calmar, max DD, total reward.

Compares against the random-action baseline rerun in-script for fair
apples-to-apples (same compute_stats math both for agent and random).

Run from project root:
    python eval/eval_synth_pretrain.py
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
from portfolio_stats import compute_stats, PortfolioStats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_DIR = PATHS.checkpoints / "synth_pretrain"

CHECKPOINTS = [
    (
        "400k_best_test",
        CKPT_DIR / "best_on_real_test" / "best_model.zip",
        CKPT_DIR / "ppo_synth_vecnormalize_400000_steps.pkl",
    ),
    (
        "600k_best_train",
        CKPT_DIR / "best_on_real_train" / "best_model.zip",
        CKPT_DIR / "ppo_synth_vecnormalize_600000_steps.pkl",
    ),
    (
        "1M_final",
        CKPT_DIR / "pretrain_final.zip",
        CKPT_DIR / "pretrain_final_vecnorm.pkl",
    ),
]

N_SEEDS = 10
REAL_EPISODE_LENGTH = 384
SYNTH_EPISODE_LENGTH = 383
INITIAL_EQUITY = 1.0


# ---------------------------------------------------------------------------
# Episode runner — returns full equity/bench curves so compute_stats can be used
# ---------------------------------------------------------------------------

def run_episode_with_curves(model, vec_env):
    """Run a deterministic episode. Returns (equity_curve, bench_curve, turnover, total_reward).
    equity/bench curves include the initial 1.0 (length n+1) for compute_stats."""
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
    return (
        np.array(equity_list),
        np.array(bench_list),
        np.array(turnover_list),
        total_reward,
    )


def run_episode_random(env):
    """Same shape of return as run_episode_with_curves, but with random actions
    on a raw (non-vec) env. Used for the random baseline."""
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
    return (
        np.array(equity_list),
        np.array(bench_list),
        np.array(turnover_list),
        total_reward,
    )


# ---------------------------------------------------------------------------
# Per-seed metric collection
# ---------------------------------------------------------------------------

def collect_metrics(equity, bench, turnover, total_reward, label="run"):
    """Compute portfolio stats for both agent and benchmark, return dict of
    medians-friendly scalars."""
    stats_agent = compute_stats(equity, label=f"{label}_agent", turnover=turnover)
    stats_bench = compute_stats(bench, label=f"{label}_bench")
    return {
        "final_equity": float(equity[-1]),
        "bench_final": float(bench[-1]),
        "alpha": float(equity[-1] - bench[-1]),
        "total_return": stats_agent.total_return,
        "cagr": stats_agent.cagr,
        "sharpe": stats_agent.sharpe,
        "sortino": stats_agent.sortino,
        "calmar": stats_agent.calmar,
        "max_dd": stats_agent.max_drawdown,
        "hit_rate": stats_agent.hit_rate,
        "avg_turnover": stats_agent.avg_turnover,
        "total_reward": total_reward,
        # Benchmark scalars for reference
        "bench_sharpe": stats_bench.sharpe,
        "bench_max_dd": stats_bench.max_drawdown,
    }


# ---------------------------------------------------------------------------
# Eval env factories
# ---------------------------------------------------------------------------

def make_real_eval_vec(features, returns, seed, vecnorm_path):
    """Vec env wrapped with loaded VecNormalize stats, for one real-data seed."""
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
    """Vec env that samples one random synth path under given seed."""
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
    """List of metric dicts -> dict of medians."""
    keys = per_seed[0].keys()
    return {k: float(np.median([d[k] for d in per_seed])) for k in keys}


def print_header():
    cols = ["eq", "alpha", "sharpe", "sortino", "calmar", "max_dd", "hit", "reward"]
    h = f"  {'label @ dataset':<28s}" + "".join(f"  {c:>7s}" for c in cols)
    print(h)
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
        v = m[key]
        s += f"  {v:>{sign}{w}.{d}f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("Synth pretrain evaluation")
    print("=" * 90)

    # ---------- Verify checkpoint files exist ----------
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
        print(f"\n{len(set(missing))} checkpoint(s) have missing files. Aborting.")
        sys.exit(1)

    # ---------- Load datasets ----------
    print("\nLoading datasets...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  real train: {feats_train.shape}")
    print(f"  real test:  {feats_test.shape}")
    print(f"  synth pool: {pool['features'].shape}")

    # ---------- Random baseline (rerun in-script) ----------
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
    print("=" * 90)
    print("RANDOM BASELINE")
    print("=" * 90)
    print_header()
    for ds in ["real_train", "real_test", "synth"]:
        print_row(f"random @ {ds}", rand_results[ds])

    # ---------- Eval each checkpoint on each dataset ----------
    all_ckpt_results = {}
    for ckpt_label, model_path, vecnorm_path in CHECKPOINTS:
        print()
        print("=" * 90)
        print(f"CHECKPOINT: {ckpt_label}")
        print(f"  model:   {model_path.name}")
        print(f"  vecnorm: {vecnorm_path.name}")
        print("=" * 90)

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

        all_ckpt_results[ckpt_label] = ds_results

    # ---------- Summary table: alpha vs random on real_test ----------
    print()
    print("=" * 90)
    print("HEADLINE: alpha vs random on real_test (the metric that matters)")
    print("=" * 90)
    rand_test_alpha = rand_results["real_test"]["alpha"]
    rand_test_eq = rand_results["real_test"]["final_equity"]
    print(f"  random:                eq={rand_test_eq:.3f}  alpha={rand_test_alpha:+.3f}")
    for ckpt_label, ds_results in all_ckpt_results.items():
        m = ds_results["real_test"]
        delta_alpha = m["alpha"] - rand_test_alpha
        delta_eq = m["final_equity"] - rand_test_eq
        print(f"  {ckpt_label:<22s} eq={m['final_equity']:.3f}  "
              f"alpha={m['alpha']:+.3f}  "
              f"Δeq vs rand={delta_eq:+.3f}  "
              f"Δalpha vs rand={delta_alpha:+.3f}")

    print()
    print("=" * 90)
    print("Done.")
    print("=" * 90)


if __name__ == "__main__":
    main()