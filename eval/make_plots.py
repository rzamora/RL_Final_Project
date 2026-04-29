"""
eval/make_plots.py

Generates figures for the final report.

Outputs (all to reports/figures/):
  fig01_equity_curves_test.png       Agent vs benchmark vs random on real_test
  fig02_drawdown_test.png            Drawdown comparison on real_test
  fig03_metrics_bar.png              Bar chart: equity, Sharpe, max_dd across checkpoints
  fig04_synth_pretrain_curve.png     Eval reward over synth pretrain steps
  fig05_finetune_comparison.png      Light vs heavy: real_test reward over fine-tune steps
  fig06_regime_breakdown.png         Per-regime returns/allocations for chosen agent

The chosen "headline" agent is light_100k_best_test.

Run from project root:
    python eval/make_plots.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from project_config import PATHS
from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    LowLevelPortfolioEnv,
    PortfolioCore,
    process_raw_df,
)
from portfolio_stats import compute_stats


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Headline agent for the report
HEADLINE_LABEL = "light_100k"
HEADLINE_MODEL = PATHS.checkpoints / "finetune_light" / "best_on_real_test" / "best_model.zip"
HEADLINE_VECNORM = (PATHS.checkpoints / "finetune_light"
                     / "ppo_finetune_light_vecnormalize_100000_steps.pkl")

# All checkpoints for the bar chart
CKPTS_FOR_BARS = [
    ("synth_600k", PATHS.checkpoints / "synth_pretrain" / "best_on_real_train" / "best_model.zip",
     PATHS.checkpoints / "synth_pretrain" / "ppo_synth_vecnormalize_600000_steps.pkl"),
    ("light_100k", PATHS.checkpoints / "finetune_light" / "best_on_real_test" / "best_model.zip",
     PATHS.checkpoints / "finetune_light" / "ppo_finetune_light_vecnormalize_100000_steps.pkl"),
    ("light_200k", PATHS.checkpoints / "finetune_light" / "best_on_real_train" / "best_model.zip",
     PATHS.checkpoints / "finetune_light" / "ppo_finetune_light_vecnormalize_200000_steps.pkl"),
    ("heavy_200k", PATHS.checkpoints / "finetune_heavy" / "best_on_real_test" / "best_model.zip",
     PATHS.checkpoints / "finetune_heavy" / "ppo_finetune_heavy_vecnormalize_200000_steps.pkl"),
    ("heavy_500k", PATHS.checkpoints / "finetune_heavy" / "best_on_real_train" / "best_model.zip",
     PATHS.checkpoints / "finetune_heavy" / "ppo_finetune_heavy_vecnormalize_500000_steps.pkl"),
]

REAL_EPISODE_LENGTH = 384
N_RANDOM_SEEDS = 10
INITIAL_EQUITY = 1.0

PRICE_COLS = ["NVDA_close", "AMD_close", "SMH_close", "TLT_close"]
REGIME_COLS = ["regime_prob_Bull", "regime_prob_Bear",
               "regime_prob_SevereBear", "regime_prob_Crisis"]
REGIME_NAMES = ["Bull", "Bear", "SevereBear", "Crisis"]
REGIME_COLORS = {"Bull": "#4daf4a", "Bear": "#ff7f00",
                 "SevereBear": "#e41a1c", "Crisis": "#984ea3"}

# Style
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Episode runners with fixed-window option
# ---------------------------------------------------------------------------

class FixedWindowCore(PortfolioCore):
    """PortfolioCore that always starts at t_start=0 instead of random."""
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.t_start = 0
        self.t_end = self.cfg.episode_length
        self.t = 0
        # Re-seed equity windows since t was changed
        self.equity = self.cfg.initial_equity
        self.bench_equity = self.cfg.initial_equity
        self.excess_dd_short = 0.0
        self.excess_dd_long = 0.0
        self.quarterly_excess = 0.0
        self.quarterly_bench = 0.0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self._init_windows()
        self.equity_window_short.append(self.equity)
        self.equity_window_long.append(self.equity)
        self.bench_equity_window_short.append(self.bench_equity)
        self.bench_equity_window_long.append(self.bench_equity)


def run_agent_fixed(model, vecnorm_path, features, returns):
    """Single deterministic episode starting at day 0 of features. Returns dict
    with equity, bench, weights, daily_returns arrays."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = FixedWindowCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        return LowLevelPortfolioEnv(core)

    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False

    obs = vec.reset()
    equity = [INITIAL_EQUITY]
    bench = [INITIAL_EQUITY]
    weights_history = []
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec.step(action)
        info = infos[0]
        equity.append(info["equity"])
        bench.append(info["bench_equity"])
        weights_history.append(info["weights"])
        rewards.append(float(reward[0]))
        done = bool(dones[0])
    vec.close()

    equity = np.array(equity)
    bench = np.array(bench)
    weights_history = np.array(weights_history)  # (T, n_assets)
    daily_returns = equity[1:] / equity[:-1] - 1.0
    bench_returns = bench[1:] / bench[:-1] - 1.0
    return {
        "equity": equity,
        "bench": bench,
        "weights": weights_history,
        "daily_returns": daily_returns,
        "bench_returns": bench_returns,
        "rewards": np.array(rewards),
    }


def run_random_fixed(features, returns, seed):
    """Single random-action episode starting at day 0 of features."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)
    core = FixedWindowCore(features, returns, cfg=cfg, rng=np.random.default_rng(seed))
    env = LowLevelPortfolioEnv(core)
    obs, _ = env.reset(seed=seed)
    equity = [INITIAL_EQUITY]
    bench = [INITIAL_EQUITY]
    daily_returns = []
    np.random.seed(seed)  # ensure action_space.sample uses this seed
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        action = rng.uniform(-1.0, 1.0, size=core.n_assets).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)
        equity.append(info["equity"])
        bench.append(info["bench_equity"])
        done = term or trunc
    return np.array(equity), np.array(bench)


def aggregate_seeded(model, vecnorm_path, features, returns, seeds):
    """Random-window seeded eval across seeds. Used for the bar chart only."""
    metrics = []
    for s in seeds:
        cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

        def _init(seed=s):
            core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(seed))
            return LowLevelPortfolioEnv(core)

        vec = DummyVecEnv([_init])
        vec = VecNormalize.load(str(vecnorm_path), vec)
        vec.training = False
        vec.norm_reward = False
        obs = vec.reset()
        equity = [INITIAL_EQUITY]
        bench = [INITIAL_EQUITY]
        turnover = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec.step(action)
            info = infos[0]
            equity.append(info["equity"])
            bench.append(info["bench_equity"])
            turnover.append(info["turnover"])
            done = bool(dones[0])
        vec.close()
        equity = np.array(equity)
        stats = compute_stats(equity, label="agent", turnover=np.array(turnover))
        metrics.append({"final_equity": equity[-1], "sharpe": stats.sharpe,
                         "max_dd": stats.max_drawdown})
    return metrics


def aggregate_random_seeded(features, returns, seeds):
    metrics = []
    for s in seeds:
        cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(s))
        env = LowLevelPortfolioEnv(core)
        obs, _ = env.reset(seed=s)
        equity = [INITIAL_EQUITY]
        rng = np.random.default_rng(s)
        done = False
        while not done:
            action = rng.uniform(-1.0, 1.0, size=core.n_assets).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            equity.append(info["equity"])
            done = term or trunc
        equity = np.array(equity)
        stats = compute_stats(equity, label="random")
        metrics.append({"final_equity": equity[-1], "sharpe": stats.sharpe,
                         "max_dd": stats.max_drawdown})
    return metrics


# ---------------------------------------------------------------------------
# Plot 1 — equity curves on test
# ---------------------------------------------------------------------------

def plot_equity_curves(test_dates, agent_run, random_runs):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Random: all seeds as faint lines + median
    rand_arr = np.stack([r[0] for r in random_runs])  # (n_seeds, T+1)
    rand_median = np.median(rand_arr, axis=0)
    for r_eq, _ in random_runs:
        ax.plot(test_dates, r_eq, color="gray", alpha=0.25, linewidth=0.7)
    ax.plot(test_dates, rand_median, color="gray", linewidth=2.0,
            label=f"Random (median, n={len(random_runs)})")

    # Benchmark (same for all runs since it's deterministic on a fixed window)
    ax.plot(test_dates, agent_run["bench"], color="C2", linewidth=2.0,
            label="Equal-weight benchmark (NVDA/AMD/SMH)")

    # Agent
    ax.plot(test_dates, agent_run["equity"], color="C0", linewidth=2.2,
            label=f"Agent ({HEADLINE_LABEL})")

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_title("Equity curves on real test set (2023 onward, fixed window)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    out = FIG_DIR / "fig01_equity_curves_test.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 2 — drawdown on test
# ---------------------------------------------------------------------------

def plot_drawdowns(test_dates, agent_run, random_runs):
    def dd(eq):
        peak = np.maximum.accumulate(eq)
        return 1.0 - eq / peak

    fig, ax = plt.subplots(figsize=(10, 4))

    rand_dds = np.stack([dd(r[0]) for r in random_runs])
    rand_median_dd = np.median(rand_dds, axis=0)
    for r_eq, _ in random_runs:
        ax.fill_between(test_dates, 0, -dd(r_eq), color="gray", alpha=0.08)
    ax.plot(test_dates, -rand_median_dd, color="gray", linewidth=1.5,
            label=f"Random (median, n={len(random_runs)})")

    ax.plot(test_dates, -dd(agent_run["bench"]), color="C2", linewidth=1.8,
            label="Equal-weight benchmark")
    ax.plot(test_dates, -dd(agent_run["equity"]), color="C0", linewidth=2.0,
            label=f"Agent ({HEADLINE_LABEL})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (negative)")
    ax.set_title("Drawdown over time on real test set")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    out = FIG_DIR / "fig02_drawdown_test.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 3 — bar chart of metrics across checkpoints (real_test)
# ---------------------------------------------------------------------------

def plot_metrics_bars(feats_test, rets_test):
    seeds = list(range(N_RANDOM_SEEDS))
    rows = []

    # Random baseline
    rand = aggregate_random_seeded(feats_test, rets_test, seeds)
    rows.append(("random",
                  np.median([m["final_equity"] for m in rand]),
                  np.median([m["sharpe"] for m in rand]),
                  np.median([m["max_dd"] for m in rand])))

    # Each checkpoint
    for label, model_p, vecnorm_p in CKPTS_FOR_BARS:
        model = PPO.load(str(model_p), device="cpu")
        ms = aggregate_seeded(model, vecnorm_p, feats_test, rets_test, seeds)
        rows.append((label,
                      np.median([m["final_equity"] for m in ms]),
                      np.median([m["sharpe"] for m in ms]),
                      np.median([m["max_dd"] for m in ms])))

    labels = [r[0] for r in rows]
    eq = [r[1] for r in rows]
    sh = [r[2] for r in rows]
    dd = [r[3] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#888888"] + ["C0"] * (len(rows) - 1)
    # Highlight headline
    headline_idx = labels.index("light_100k")
    colors[headline_idx] = "#d62728"

    for ax, vals, title, ylim in [
        (axes[0], eq, "Final equity (median)", None),
        (axes[1], sh, "Sharpe (median)", None),
        (axes[2], dd, "Max drawdown (median)", None),
    ]:
        ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if title.startswith("Final") else f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    axes[0].axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    fig.suptitle("Real test performance across checkpoints (10 seeds, median)",
                  fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig03_metrics_bar.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 4 — synth pretrain training curve (eval reward vs steps)
# ---------------------------------------------------------------------------

def plot_synth_pretrain_curve():
    """Reads the eval data the EvalCallback saved during synth pretrain."""
    eval_path = PATHS.tb_logs / "synth_pretrain"
    train_npz = eval_path / "eval_real_train" / "evaluations.npz"
    test_npz = eval_path / "eval_real_test" / "evaluations.npz"

    if not train_npz.exists() or not test_npz.exists():
        print(f"  SKIP: eval npz files not found at {eval_path}")
        return

    train_data = np.load(train_npz)
    test_data = np.load(test_npz)

    # evaluations.npz fields: timesteps, results (n_evals, n_eval_episodes)
    train_steps = train_data["timesteps"]
    test_steps = test_data["timesteps"]
    train_means = train_data["results"].mean(axis=1)
    test_means = test_data["results"].mean(axis=1)
    train_stds = train_data["results"].std(axis=1)
    test_stds = test_data["results"].std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train_steps, train_means, "C0-o", label="real train eval", linewidth=1.8)
    ax.fill_between(train_steps, train_means - train_stds, train_means + train_stds,
                     color="C0", alpha=0.15)
    ax.plot(test_steps, test_means, "C3-s", label="real test eval", linewidth=1.8)
    ax.fill_between(test_steps, test_means - test_stds, test_means + test_stds,
                     color="C3", alpha=0.15)

    ax.axhline(-31.5, color="gray", linestyle="--", linewidth=1.0,
                label="random baseline (≈ -31.5)")
    ax.axvline(600_000, color="black", linestyle=":", linewidth=1.0,
                label="600k checkpoint (chosen)")

    ax.set_xlabel("Synth pretrain step")
    ax.set_ylabel("Eval mean reward (10 episodes)")
    ax.set_title("Synth pretrain: eval reward over training steps")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    out = FIG_DIR / "fig04_synth_pretrain_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 5 — light vs heavy fine-tune comparison
# ---------------------------------------------------------------------------

def plot_finetune_comparison():
    light_test = PATHS.tb_logs / "finetune_light" / "eval_real_test" / "evaluations.npz"
    heavy_test = PATHS.tb_logs / "finetune_heavy" / "eval_real_test" / "evaluations.npz"
    light_train = PATHS.tb_logs / "finetune_light" / "eval_real_train" / "evaluations.npz"
    heavy_train = PATHS.tb_logs / "finetune_heavy" / "eval_real_train" / "evaluations.npz"

    if not all(p.exists() for p in [light_test, heavy_test, light_train, heavy_train]):
        print(f"  SKIP: fine-tune eval npz files missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, (label_tag, color) in zip(axes, [("real_train", "C0"), ("real_test", "C3")]):
        light = np.load(PATHS.tb_logs / "finetune_light" / f"eval_{label_tag}" / "evaluations.npz")
        heavy = np.load(PATHS.tb_logs / "finetune_heavy" / f"eval_{label_tag}" / "evaluations.npz")
        ax.plot(light["timesteps"], light["results"].mean(axis=1),
                "C0-o", label="light fine-tune", linewidth=1.8)
        ax.plot(heavy["timesteps"], heavy["results"].mean(axis=1),
                "C1-s", label="heavy fine-tune", linewidth=1.8)
        ax.set_xlabel("Fine-tune step")
        ax.set_title(f"Eval reward on {label_tag.replace('_', ' ')}")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Eval mean reward")
    fig.suptitle("Fine-tune comparison: light vs heavy", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig05_finetune_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 6 — regime breakdown
# ---------------------------------------------------------------------------

def plot_regime_breakdown(test_df, agent_run):
    """Partition agent's daily behavior by regime label. Layout chosen to
    make regime-blindness of the LL policy visible:
      (a) regime distribution on test
      (b) per-regime mean asset weights — should look flat across regimes
      (c) per-regime alpha (agent_ret - bench_ret)
    """
    T = len(agent_run["daily_returns"])
    regimes = test_df[REGIME_COLS].iloc[1:T+1].to_numpy()
    regime_idx = regimes.argmax(axis=1)

    agent_rets = agent_run["daily_returns"]
    bench_rets = agent_run["bench_returns"]
    weights = agent_run["weights"]
    asset_names = ["NVDA", "AMD", "SMH", "TLT"]
    asset_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Regime distribution
    counts = np.bincount(regime_idx, minlength=4)
    shares = counts / T
    axes[0].bar(REGIME_NAMES, shares,
                 color=[REGIME_COLORS[n] for n in REGIME_NAMES])
    axes[0].set_ylabel("Fraction of test period")
    axes[0].set_title("Regime distribution on real test")
    for i, s in enumerate(shares):
        axes[0].text(i, s, f"{s:.1%}", ha="center", va="bottom", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    # (b) Per-regime mean asset weights — grouped bars, flat = regime-blind
    x = np.arange(4)  # one group per regime
    width = 0.20
    for asset_i, asset_name in enumerate(asset_names):
        means = [weights[regime_idx == i, asset_i].mean() if (regime_idx == i).any() else 0
                  for i in range(4)]
        axes[1].bar(x + (asset_i - 1.5) * width, means, width,
                     label=asset_name, color=asset_colors[asset_i])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(REGIME_NAMES)
    axes[1].set_ylabel("Mean weight")
    axes[1].set_title("Per-regime mean asset weights\n(flat across regimes = regime-blind LL)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend(ncol=2, fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.3, axis="y")

    # (c) Per-regime alpha (agent - bench mean daily return)
    agent_means = np.array(
        [agent_rets[regime_idx == i].mean() if (regime_idx == i).any() else 0
         for i in range(4)]
    )
    bench_means = np.array(
        [bench_rets[regime_idx == i].mean() if (regime_idx == i).any() else 0
         for i in range(4)]
    )
    alpha_per_regime = (agent_means - bench_means) * 100  # percent
    bar_colors = ["#4daf4a" if a >= 0 else "#e41a1c" for a in alpha_per_regime]
    axes[2].bar(REGIME_NAMES, alpha_per_regime, color=bar_colors)
    axes[2].set_ylabel("Mean daily alpha (%)")
    axes[2].set_title("Per-regime alpha (agent − benchmark)")
    axes[2].axhline(0, color="black", linewidth=0.5)
    for i, a in enumerate(alpha_per_regime):
        axes[2].text(i, a, f"{a:+.3f}", ha="center",
                      va="bottom" if a >= 0 else "top", fontsize=9)
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Regime-conditional analysis on real test set", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig06_regime_breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")

    print("\n  Regime breakdown summary:")
    for i, name in enumerate(REGIME_NAMES):
        if not (regime_idx == i).any():
            continue
        n = (regime_idx == i).sum()
        a = agent_means[i] * 100
        b = bench_means[i] * 100
        mw = weights[regime_idx == i].mean(axis=0)
        print(f"    {name:<11s}  n={n:>3d}  share={shares[i]:.1%}  "
              f"alpha={a-b:+.3f}%  weights=[{mw[0]:+.2f},{mw[1]:+.2f},"
              f"{mw[2]:+.2f},{mw[3]:+.2f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Output dir: {FIG_DIR}")
    print()

    # Load datasets
    print("Loading data...")
    test_df = pd.read_csv(PATHS.test_csv)
    feats_test, rets_test, _ = process_raw_df(test_df)
    test_dates = pd.to_datetime(test_df["date"])

    # Headline agent: one fixed-window run on test
    print(f"\nRunning headline agent ({HEADLINE_LABEL}) on test, fixed window...")
    headline_model = PPO.load(str(HEADLINE_MODEL), device="cpu")
    agent_run = run_agent_fixed(headline_model, HEADLINE_VECNORM, feats_test, rets_test)
    print(f"  final equity: {agent_run['equity'][-1]:.4f}")
    print(f"  bench final:  {agent_run['bench'][-1]:.4f}")

    # Random runs on test, fixed window, multiple seeds
    print(f"\nRunning {N_RANDOM_SEEDS} random seeds on test, fixed window...")
    random_runs = []
    for s in range(N_RANDOM_SEEDS):
        eq, bench = run_random_fixed(feats_test, rets_test, s)
        random_runs.append((eq, bench))

    # Slice dates to match equity curve length (T+1 = REAL_EPISODE_LENGTH+1)
    dates_for_plot = test_dates.iloc[:REAL_EPISODE_LENGTH + 1].reset_index(drop=True)

    # Plots
    print("\nGenerating plots...")
    plot_equity_curves(dates_for_plot, agent_run, random_runs)
    plot_drawdowns(dates_for_plot, agent_run, random_runs)
    plot_metrics_bars(feats_test, rets_test)
    plot_synth_pretrain_curve()
    plot_finetune_comparison()
    plot_regime_breakdown(test_df, agent_run)

    print(f"\nDone. {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()