"""
Portfolio statistics + benchmark comparison.

Compares an agent's equity curve to an equal-weight (NVDA, AMD, SMH)
buy-and-rebalance benchmark on the same date range.

Usage:
    from portfolio_stats import (
        compute_stats,
        run_equal_weight_benchmark,
        run_agent_rollout,
        compare,
    )

    stats_agent, eq_agent = run_agent_rollout(model, core)
    stats_bench, eq_bench = run_equal_weight_benchmark(returns, dates)
    compare(stats_agent, stats_bench, eq_agent, eq_bench, dates)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ============================================================
# Statistics container
# ============================================================

@dataclass
class PortfolioStats:
    label: str
    total_return: float           # final equity / initial - 1
    cagr: float                   # annualized geometric return
    vol_ann: float                # annualized volatility of daily returns
    sharpe: float                 # (mean - rf) / std, annualized; rf=0 unless overridden
    sortino: float                # mean / downside_std, annualized
    max_drawdown: float           # most negative peak-to-trough as a positive fraction
    avg_drawdown: float           # mean drawdown (always >= 0)
    calmar: float                 # cagr / max_drawdown
    var_95: float                 # 5th percentile of daily returns (negative number)
    cvar_95: float                # mean of returns below the 5th percentile
    hit_rate: float               # fraction of days with positive return
    skew: float
    kurtosis: float               # excess kurtosis
    avg_turnover: float           # mean daily turnover (sum of |Δw|), 0 if not tracked
    n_days: int

    def to_dict(self):
        return asdict(self)


# ============================================================
# Core stats
# ============================================================

def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Per-step drawdown as a non-negative fraction."""
    peak = np.maximum.accumulate(equity)
    return 1.0 - equity / peak


def compute_stats(
    equity: np.ndarray,
    label: str,
    rf_annual: float = 0.0,
    turnover: Optional[np.ndarray] = None,
) -> PortfolioStats:
    """
    Compute summary statistics from an equity curve.

    Parameters
    ----------
    equity : array of length n+1
        Equity curve. equity[0] is initial capital, equity[-1] is final.
        We compute returns as equity[1:] / equity[:-1] - 1.
    label : str
        Display name (e.g. "Agent", "EW(NVDA,AMD,SMH)").
    rf_annual : float
        Annual risk-free rate. Daily rate = (1 + rf_annual)^(1/252) - 1.
    turnover : optional array of length n
        Per-step turnover (sum of |Δw|). Only used to report mean turnover.
    """
    equity = np.asarray(equity, dtype=np.float64)
    if equity.ndim != 1 or len(equity) < 2:
        raise ValueError("equity must be a 1-D array of length >= 2")

    rets = equity[1:] / equity[:-1] - 1.0
    n = len(rets)

    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = rets - rf_daily

    total_return = float(equity[-1] / equity[0] - 1.0)
    years = n / TRADING_DAYS
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    vol_ann = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if n > 1 and excess.std(ddof=1) > 0
        else 0.0
    )

    downside = excess[excess < 0]
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = float(excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
    else:
        sortino = 0.0

    dd = _drawdown_series(equity)
    max_dd = float(dd.max())
    avg_dd = float(dd.mean())
    calmar = float(cagr / max_dd) if max_dd > 1e-8 else 0.0

    var_95 = float(np.quantile(rets, 0.05))
    tail = rets[rets <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95

    hit_rate = float((rets > 0).mean())
    skew = float(_moment_skew(rets))
    kurt = float(_moment_excess_kurt(rets))

    avg_to = float(np.asarray(turnover).mean()) if turnover is not None else 0.0

    return PortfolioStats(
        label=label,
        total_return=total_return,
        cagr=cagr,
        vol_ann=vol_ann,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        calmar=calmar,
        var_95=var_95,
        cvar_95=cvar_95,
        hit_rate=hit_rate,
        skew=skew,
        kurtosis=kurt,
        avg_turnover=avg_to,
        n_days=n,
    )


def _moment_skew(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    return ((x - m) ** 3).mean() / (s**3 + 1e-12)


def _moment_excess_kurt(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    return ((x - m) ** 4).mean() / (s**4 + 1e-12) - 3.0


# ============================================================
# Equal-weight benchmark
# ============================================================

def run_equal_weight_benchmark(
    returns: np.ndarray,
    weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3, 0.0),
    initial_equity: float = 1.0,
    label: str = "EW(NVDA,AMD,SMH)",
    daily_rebalance: bool = True,
) -> tuple[PortfolioStats, np.ndarray]:
    """
    Run an equal-weight benchmark on the asset return panel.

    By default holds 1/3 each in NVDA, AMD, SMH and 0 in TLT, rebalanced daily.

    Parameters
    ----------
    returns : (n_days, 4) array
        Daily returns for [NVDA, AMD, SMH, TLT] in fractional form (e.g. 0.01 = 1%).
    weights : sequence of 4 floats
        Target weights per asset. Default: 1/3, 1/3, 1/3, 0.
    initial_equity : float
    label : str
    daily_rebalance : bool
        If True, rebalance back to target weights every day (drift-free).
        If False, hold weights fixed and let them drift (buy-and-hold).

    Returns
    -------
    stats : PortfolioStats
    equity : array of shape (n_days + 1,)
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[1] != 4:
        raise ValueError(f"returns must be (n_days, 4), got {returns.shape}")

    w = np.asarray(weights, dtype=np.float64)
    if not np.isclose(w.sum(), 1.0):
        # allow non-unit if intentional, but warn via label
        pass

    n = returns.shape[0]
    equity = np.zeros(n + 1, dtype=np.float64)
    equity[0] = initial_equity
    turnover = np.zeros(n, dtype=np.float64)

    if daily_rebalance:
        port_rets = returns @ w
        equity[1:] = initial_equity * np.cumprod(1.0 + port_rets)
        # turnover from daily rebalance: at each day, weights drift by r_i - r_p
        # then snap back to w. Cost = sum |Δw| ≈ sum |w_i (r_i - r_p) / (1 + r_p)|.
        drift = (returns - port_rets[:, None]) * w[None, :]
        turnover = np.sum(np.abs(drift), axis=1)
    else:
        # buy and hold: weights drift, never rebalance
        cur_w = w.copy()
        for t in range(n):
            r = returns[t]
            wealth_relative = 1.0 + cur_w @ r
            equity[t + 1] = equity[t] * wealth_relative
            new_w = cur_w * (1.0 + r) / wealth_relative
            turnover[t] = float(np.sum(np.abs(new_w - cur_w)))
            cur_w = new_w

    stats = compute_stats(equity, label=label, turnover=turnover)
    return stats, equity


# ============================================================
# Agent rollout
# ============================================================

def run_agent_rollout(
    model,
    env,
    deterministic: bool = True,
    label: str = "Agent",
) -> tuple[PortfolioStats, np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out a trained policy on an env (HighLevelPortfolioEnv recommended)
    and collect the equity curve.

    Parameters
    ----------
    model : SB3 model with `.predict(obs, deterministic=...)`
    env : gym env that returns info dict with 'equity', 'turnover', 'weights'
    deterministic : bool

    Returns
    -------
    stats : PortfolioStats
    equity : (T+1,)
    weights_history : (T, n_assets)
    turnover_history : (T,)
    """
    obs, _ = env.reset()
    done = False

    equity_list = [env.core.equity if hasattr(env, "core") else 1.0]
    weights_list = []
    turnover_list = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity_list.append(info["equity"])
        weights_list.append(info["weights"])
        turnover_list.append(info["turnover"])

    equity = np.array(equity_list)
    weights = np.array(weights_list)
    turnover = np.array(turnover_list)
    stats = compute_stats(equity, label=label, turnover=turnover)
    return stats, equity, weights, turnover


# ============================================================
# Comparison report
# ============================================================

def compare(
    stats_a: PortfolioStats,
    stats_b: PortfolioStats,
    equity_a: Optional[np.ndarray] = None,
    equity_b: Optional[np.ndarray] = None,
    dates: Optional[Sequence] = None,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Side-by-side comparison of two portfolios. Returns a DataFrame and
    (optionally) prints a formatted table.
    """
    rows = [
        ("Total return",        "pct",  "total_return"),
        ("CAGR",                "pct",  "cagr"),
        ("Annualized vol",      "pct",  "vol_ann"),
        ("Sharpe ratio",        "num",  "sharpe"),
        ("Sortino ratio",       "num",  "sortino"),
        ("Max drawdown",        "pct",  "max_drawdown"),
        ("Avg drawdown",        "pct",  "avg_drawdown"),
        ("Calmar ratio",        "num",  "calmar"),
        ("VaR (95%, daily)",    "pct",  "var_95"),
        ("CVaR (95%, daily)",   "pct",  "cvar_95"),
        ("Hit rate",            "pct",  "hit_rate"),
        ("Skewness",            "num",  "skew"),
        ("Excess kurtosis",     "num",  "kurtosis"),
        ("Avg daily turnover",  "pct",  "avg_turnover"),
        ("# days",              "int",  "n_days"),
    ]

    a, b = stats_a.to_dict(), stats_b.to_dict()
    df = pd.DataFrame(
        {
            "metric":     [r[0] for r in rows],
            stats_a.label: [a[r[2]] for r in rows],
            stats_b.label: [b[r[2]] for r in rows],
        }
    )
    df["difference"] = df[stats_a.label] - df[stats_b.label]

    if print_table:
        print(_format_table(df, rows, stats_a.label, stats_b.label))

    return df


def _format_table(df, rows, label_a, label_b):
    out = []
    width_metric = max(len(r[0]) for r in rows)
    width_val = 14
    header = (
        f"{'Metric':<{width_metric}}  "
        f"{label_a:>{width_val}}  "
        f"{label_b:>{width_val}}  "
        f"{'Diff (A-B)':>{width_val}}"
    )
    out.append(header)
    out.append("-" * len(header))
    for i, (name, kind, key) in enumerate(rows):
        va = df[label_a].iloc[i]
        vb = df[label_b].iloc[i]
        diff = df["difference"].iloc[i]
        if kind == "pct":
            sa, sb, sd = f"{va:>{width_val-2}.2%}", f"{vb:>{width_val-2}.2%}", f"{diff:>+{width_val-2}.2%}"
        elif kind == "int":
            sa, sb, sd = f"{int(va):>{width_val}d}", f"{int(vb):>{width_val}d}", f"{int(diff):>+{width_val}d}"
        else:
            sa, sb, sd = f"{va:>{width_val}.3f}", f"{vb:>{width_val}.3f}", f"{diff:>+{width_val}.3f}"
        out.append(f"{name:<{width_metric}}  {sa}  {sb}  {sd}")
    return "\n".join(out)


# ============================================================
# Plotting (optional — only if matplotlib is available)
# ============================================================

def plot_equity_curves(
    curves: dict,
    dates: Optional[Sequence] = None,
    title: str = "Equity curves",
    log_y: bool = True,
    show_drawdown: bool = True,
    save_path: Optional[str] = None,
):
    """
    curves: dict[label] = equity array (length T+1)
    """
    import matplotlib.pyplot as plt

    n_panels = 2 if show_drawdown else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11, 7 if show_drawdown else 4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]} if show_drawdown else None,
    )
    if n_panels == 1:
        axes = [axes]

    for label, eq in curves.items():
        x = pd.to_datetime(dates[: len(eq)]) if dates is not None else np.arange(len(eq))
        axes[0].plot(x, eq, label=label, linewidth=1.5)

    axes[0].set_title(title)
    axes[0].set_ylabel("Equity (log scale)" if log_y else "Equity")
    if log_y:
        axes[0].set_yscale("log")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    if show_drawdown:
        for label, eq in curves.items():
            x = pd.to_datetime(dates[: len(eq)]) if dates is not None else np.arange(len(eq))
            dd = _drawdown_series(eq)
            axes[1].fill_between(x, 0, -dd, alpha=0.4, label=label)
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


# ============================================================
# Convenience: end-to-end on the test CSV
# ============================================================

def evaluate_on_csv(
    csv_path: str,
    model=None,
    env_factory=None,
    benchmark_weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3, 0.0),
    save_dir: Optional[str] = None,
):
    """
    Convenience wrapper that:
      1. loads the CSV
      2. runs the equal-weight benchmark on its returns panel
      3. (if model+env_factory provided) runs the agent
      4. prints the comparison and (if save_dir given) saves the equity-curve plot

    env_factory: callable that takes (features, returns) and returns a gym env.

    Note: the *_close columns are dropped from the features fed to the env
    because absolute price levels leak temporal position into the policy.
    The prices are still used to compute returns for the simulator.
    """
    df = pd.read_csv(csv_path)
    dates = df["date"].values
    price_cols = ["NVDA_close", "AMD_close", "SMH_close", "TLT_close"]
    returns_panel = df[price_cols].pct_change().fillna(0.0).to_numpy(dtype=np.float64)
    # drop the first row (zero return) so the benchmark equity curve is well-defined
    returns_panel = returns_panel[1:]
    dates = dates[1:]

    bench_stats, bench_eq = run_equal_weight_benchmark(
        returns_panel, weights=benchmark_weights, daily_rebalance=True
    )

    if model is None or env_factory is None:
        print(_format_single(bench_stats))
        return {"benchmark": (bench_stats, bench_eq)}

    # Drop date AND close columns from features — closes are absolute price
    # levels that leak the calendar position into the policy. Match this with
    # the synthetic pool loader and process_raw_df.
    feats_df = df.drop(columns=["date"] + price_cols).iloc[1:]  # align with returns_panel
    features = feats_df.to_numpy(dtype=np.float32)
    env = env_factory(features, returns_panel.astype(np.float32))
    agent_stats, agent_eq, _, _ = run_agent_rollout(model, env)

    compare(agent_stats, bench_stats, agent_eq, bench_eq, dates=dates)

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plot_equity_curves(
            {"Agent": agent_eq, "EW Benchmark": bench_eq},
            dates=dates,
            save_path=f"{save_dir}/equity_curves.png",
        )

    return {
        "agent":     (agent_stats, agent_eq),
        "benchmark": (bench_stats, bench_eq),
    }


def _format_single(stats: PortfolioStats) -> str:
    lines = [f"--- {stats.label} ---"]
    for k, v in stats.to_dict().items():
        if k == "label":
            continue
        if isinstance(v, float) and abs(v) < 5:
            lines.append(f"  {k:<20s} {v:>10.4f}")
        else:
            lines.append(f"  {k:<20s} {v}")
    return "\n".join(lines)


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    # Synthetic smoke test — no model, no env, just verify the math
    rng = np.random.default_rng(0)
    n_days = 500
    fake_returns = rng.normal(0.0005, 0.015, size=(n_days, 4)).astype(np.float64)
    fake_returns[:, 3] *= 0.5  # TLT lower vol
    fake_returns[:, 3] -= 0.0003  # mild negative drift

    stats, eq = run_equal_weight_benchmark(fake_returns)
    print(_format_single(stats))
    print()
    print(f"Final equity: {eq[-1]:.4f}")
    print(f"Days: {len(eq) - 1}")
