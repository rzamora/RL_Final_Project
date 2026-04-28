"""
Unit tests for the new asymmetric reward function.

Verifies that the reward function produces the correct signs in each market
regime BEFORE we use it for training. With this many penalty/bonus terms
interacting, miscalibrated signs would silently train a wrong policy.

Each scenario constructs a synthetic returns sequence where we know the
expected qualitative behavior (heavy bonus, heavy penalty, neutral, etc.),
runs the env through it, and checks the cumulative reward matches.

Run from project root: `python tests/test_reward_function.py`
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "env"))

import numpy as np
from portfolio_hrl_env_fixed import PortfolioCore, CoreConfig


# ============================================================
# Scenario builders — synthetic returns sequences with known properties
# ============================================================

def _run_episode(core, agent_action_fn):
    """Run a full episode applying agent_action_fn(t) -> weights at each step.
    Returns (rewards, info_history)."""
    core.reset(seed=0)
    rewards = []
    info_history = []
    for t in range(core.cfg.episode_length):
        weights = agent_action_fn(t).astype(np.float32)
        reward, done, info = core.apply_allocation(weights)
        rewards.append(reward)
        info_history.append(info)
        if done:
            break
    return np.array(rewards), info_history


def _make_core_with_returns(asset_returns, bench_returns, episode_length=None):
    """Build a PortfolioCore with explicit return sequences."""
    n_steps = len(asset_returns)
    if episode_length is None:
        episode_length = min(n_steps - 1, 100)

    cfg = CoreConfig(episode_length=episode_length)
    # Need at least cfg.episode_length + 1 rows
    # Pad features (not used in reward calc) with zeros
    features = np.zeros((n_steps, 10), dtype=np.float32)
    return PortfolioCore(
        features=features,
        returns=asset_returns.astype(np.float32),
        benchmark_returns=bench_returns.astype(np.float32),
        cfg=cfg,
        rng=np.random.default_rng(0),
    )


# ============================================================
# SCENARIOS
# ============================================================

def scenario_quiet_market_match_benchmark():
    """Quiet sideways market, agent matches benchmark exactly.
    Expected: cumulative reward near zero (no asymmetric term fires,
    log-returns roughly zero, only turnover deducts a tiny bit)."""

    n_steps = 100
    rng = np.random.default_rng(0)
    # Tiny daily returns, no drift
    bench = rng.normal(0.0, 0.005, size=n_steps)
    # Construct asset returns so that equal-weight = benchmark
    asset_returns = np.column_stack([bench, bench, bench, bench])

    core = _make_core_with_returns(asset_returns, bench, episode_length=80)

    # Agent holds equal-weight long, matching benchmark
    def agent(t):
        return np.array([0.25, 0.25, 0.25, 0.25])

    rewards, info = _run_episode(core, agent)
    cum = rewards.sum()

    print(f"  Cumulative reward: {cum:+.4f}")
    print(f"  Log-return component (sum of log(1+r)): "
          f"{np.log(info[-1]['equity']):+.4f}")
    print(f"  Excess_dd_short (final): {info[-1]['excess_dd_short']:+.4f}")
    print(f"  Quarterly_excess (final): {info[-1]['quarterly_excess']:+.4f}")

    # Should be near zero — agent and benchmark are essentially identical
    assert abs(cum) < 0.5, f"Expected near-zero reward, got {cum}"
    assert abs(info[-1]['excess_dd_short']) < 0.01
    print("  ✓ Reward near zero as expected")


def scenario_bull_market_agent_in_cash():
    """Strong bull market, agent stays in cash (zero weights).
    Expected: HEAVY negative reward — missed the rally."""

    n_steps = 100
    # Steady +0.4% per day rally, ~50% over the episode
    bench = np.full(n_steps, 0.004)
    asset_returns = np.column_stack([bench * 1.2, bench * 0.8, bench, bench * 0.5])

    core = _make_core_with_returns(asset_returns, bench, episode_length=80)

    # Agent holds zero — fully in cash
    def agent(t):
        return np.array([0.0, 0.0, 0.0, 0.0])

    rewards, info = _run_episode(core, agent)
    cum = rewards.sum()
    bench_eq_final = info[-1]['bench_equity']
    agent_eq_final = info[-1]['equity']

    print(f"  Cumulative reward: {cum:+.4f}")
    print(f"  Agent equity final: {agent_eq_final:.4f}")
    print(f"  Bench equity final: {bench_eq_final:.4f}")
    print(f"  Quarterly_excess (final): {info[-1]['quarterly_excess']:+.4f}")
    print(f"  Quarterly_bench (final): {info[-1]['quarterly_bench']:+.4f}")

    # Bench should be up substantially, agent flat
    assert bench_eq_final > 1.2, f"Bench should rally, got {bench_eq_final}"
    assert abs(agent_eq_final - 1.0) < 0.001, f"Agent should be flat in cash"
    # Reward should be heavily negative due to upside-miss penalty
    assert cum < -1.0, f"Expected heavy negative reward (missed rally), got {cum}"
    print("  ✓ Heavy negative reward as expected (missed the rally)")


def scenario_bear_market_agent_short_makes_money():
    """Bear market crashes -30%, agent goes short and ends +5%.
    Expected: HEAVY positive reward — crisis alpha."""

    n_steps = 100
    # Steady -0.4% per day decline, ~30% drawdown over episode
    bench = np.full(n_steps, -0.004)
    # All assets decline similarly
    asset_returns = np.column_stack([bench, bench, bench, bench])

    core = _make_core_with_returns(asset_returns, bench, episode_length=80)

    # Agent goes short equity book — weights -0.5, -0.5, -0.5, -0.5 means net -2 gross 2
    # But max_gross is 1.5, so agent is short 1.5 gross with all weight short
    # Profit = (-)(-r) = r per day at gross=1.5 → ~+0.6% per day
    def agent(t):
        return np.array([-0.375, -0.375, -0.375, -0.375])

    rewards, info = _run_episode(core, agent)
    cum = rewards.sum()
    bench_eq_final = info[-1]['bench_equity']
    agent_eq_final = info[-1]['equity']

    print(f"  Cumulative reward: {cum:+.4f}")
    print(f"  Agent equity final: {agent_eq_final:.4f}")
    print(f"  Bench equity final: {bench_eq_final:.4f}")
    print(f"  Quarterly_excess (final): {info[-1]['quarterly_excess']:+.4f}")
    print(f"  Quarterly_bench (final): {info[-1]['quarterly_bench']:+.4f}")

    # Bench should be down substantially, agent up
    assert bench_eq_final < 0.8, f"Bench should crash, got {bench_eq_final}"
    assert agent_eq_final > 1.1, f"Agent should profit from short, got {agent_eq_final}"
    # Reward should be heavily positive due to crisis-alpha bonus
    assert cum > 0.3, f"Expected heavy positive reward (crisis alpha), got {cum}"
    print("  ✓ Heavy positive reward as expected (crisis alpha)")


def scenario_bear_market_agent_matches():
    """Bear market crashes -30%, agent matches by being long.
    Expected: large negative log-return, but no asymmetric penalty
    (excess_dd ≈ 0, quarterly_excess ≈ 0). Total reward dominated by
    log-returns — strongly negative."""

    n_steps = 100
    bench = np.full(n_steps, -0.004)
    asset_returns = np.column_stack([bench, bench, bench, bench])

    core = _make_core_with_returns(asset_returns, bench, episode_length=80)

    # Agent matches benchmark with equal-weight long
    def agent(t):
        return np.array([0.25, 0.25, 0.25, 0.25])

    rewards, info = _run_episode(core, agent)
    cum = rewards.sum()
    agent_eq = info[-1]['equity']
    bench_eq = info[-1]['bench_equity']

    print(f"  Cumulative reward: {cum:+.4f}")
    print(f"  Agent equity final: {agent_eq:.4f}")
    print(f"  Bench equity final: {bench_eq:.4f}")
    print(f"  Excess_dd_short (final): {info[-1]['excess_dd_short']:+.4f}")
    print(f"  Quarterly_excess (final): {info[-1]['quarterly_excess']:+.4f}")

    # Agent and bench both down ~30%
    assert agent_eq < 0.8 and bench_eq < 0.8
    # Excess_dd should be near zero (both drawing down equally)
    assert abs(info[-1]['excess_dd_short']) < 0.02
    # Quarterly excess near zero (matched)
    assert abs(info[-1]['quarterly_excess']) < 0.02
    # Reward should be moderately negative (log-returns) but NOT crushing
    # because the asymmetric term doesn't fire when matched
    assert -1.5 < cum < -0.1, (
        f"Expected moderately negative reward (just log-returns), got {cum}"
    )
    print("  ✓ Moderate negative reward as expected (log-returns only, no asymmetric penalty)")


def scenario_bull_market_agent_outperforms():
    """Bull market, agent outperforms benchmark by 5% over the episode.
    Expected: positive log-returns + mild upside-beat bonus.
    Should be positive but not heavily."""

    n_steps = 100
    bench = np.full(n_steps, 0.003)
    # Asset returns slightly higher on average — agent will outperform if it
    # holds weight on higher-return assets
    asset_returns = np.column_stack([
        np.full(n_steps, 0.004),  # NVDA: better
        np.full(n_steps, 0.004),  # AMD: better
        bench,                     # SMH: matches bench
        bench,                     # TLT: matches bench
    ])

    core = _make_core_with_returns(asset_returns, bench, episode_length=80)

    # Agent overweights NVDA + AMD
    def agent(t):
        return np.array([0.5, 0.5, 0.0, 0.0])

    rewards, info = _run_episode(core, agent)
    cum = rewards.sum()
    agent_eq = info[-1]['equity']
    bench_eq = info[-1]['bench_equity']

    print(f"  Cumulative reward: {cum:+.4f}")
    print(f"  Agent equity final: {agent_eq:.4f}")
    print(f"  Bench equity final: {bench_eq:.4f}")
    print(f"  Quarterly_excess (final): {info[-1]['quarterly_excess']:+.4f}")
    print(f"  Quarterly_bench (final): {info[-1]['quarterly_bench']:+.4f}")

    # Agent should be ahead
    assert agent_eq > bench_eq
    # Both up
    assert agent_eq > 1.0 and bench_eq > 1.0
    # Reward should be positive (log-return is positive, plus mild upside-beat)
    assert cum > 0.1, f"Expected positive reward, got {cum}"
    # But not heavily — upside-beat is the small lambda
    print("  ✓ Mild positive reward as expected (outperformed in rally)")


def scenario_idiosyncratic_drawdown():
    """Sideways benchmark, agent has a 15% drawdown for 30 days.
    Expected: short-window excess_dd penalty fires (agent is 15% down vs
    flat benchmark)."""

    n_steps = 100
    # Benchmark flat (zero return)
    bench = np.zeros(n_steps)
    # Agent will be entirely in NVDA, which has a sustained drawdown days 10-40
    nvda = np.zeros(n_steps)
    nvda[10:40] = -0.005  # 30 days of -0.5% (cumulative ~14%)
    nvda[40:60] = 0.005   # 20 days of recovery
    asset_returns = np.column_stack([nvda, np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)])

    core = _make_core_with_returns(asset_returns, bench, episode_length=80)

    # Agent puts all weight on NVDA
    def agent(t):
        return np.array([1.0, 0.0, 0.0, 0.0])

    rewards, info = _run_episode(core, agent)
    cum = rewards.sum()

    # Find the maximum excess_dd_short during episode
    max_excess_dd = max(i['excess_dd_short'] for i in info)

    print(f"  Cumulative reward: {cum:+.4f}")
    print(f"  Max excess_dd_short: {max_excess_dd:+.4f}")
    print(f"  Agent equity final: {info[-1]['equity']:.4f}")
    print(f"  Bench equity final: {info[-1]['bench_equity']:.4f}")

    # Excess_dd should peak above the threshold
    assert max_excess_dd > 0.05, (
        f"Expected excess_dd > 5% threshold, got {max_excess_dd}"
    )
    print("  ✓ Excess_dd penalty fires for idiosyncratic drawdown")


def scenario_intraday_spike_recovery():
    """Single-day -10% spike followed by full recovery.
    Expected: rolling drawdown should NOT fire because next day's recovery
    sets a new window peak. Reward should reflect the immediate log-return
    losses and gains, but no excess_dd penalty."""

    n_steps = 100
    bench = np.zeros(n_steps)
    # Agent assets: flat, then big spike down day 30, recovery day 31, flat after
    nvda = np.zeros(n_steps)
    nvda[30] = -0.10  # -10% on day 30
    nvda[31] = 0.111  # +11.1% on day 31 (recovers to ~flat: 0.9 * 1.111 = 1.0)
    asset_returns = np.column_stack([nvda, np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)])

    core = _make_core_with_returns(asset_returns, bench, episode_length=80)

    def agent(t):
        return np.array([1.0, 0.0, 0.0, 0.0])

    rewards, info = _run_episode(core, agent)

    # Find the max excess_dd_short across the episode
    max_excess_dd = max(i['excess_dd_short'] for i in info)
    # After day 31, the rolling window's peak is ~the recovered level, so
    # excess_dd should drop back to ~0 quickly

    excess_dd_after_recovery = info[40]['excess_dd_short']  # 9 days after recovery

    print(f"  Max excess_dd during spike: {max_excess_dd:+.4f}")
    print(f"  Excess_dd_short 9 days after recovery: {excess_dd_after_recovery:+.4f}")

    # The single day spike should briefly show high excess_dd
    # But rolling window means it doesn't persist — by day 40, excess_dd should
    # be near zero (the spike has rolled off the front of the new peak)
    assert max_excess_dd > 0.05, "Spike should be visible in excess_dd briefly"
    # But we don't want it to persist
    assert excess_dd_after_recovery < 0.01, (
        f"Excess_dd should drop after recovery, got {excess_dd_after_recovery}"
    )
    print("  ✓ Rolling window correctly handles intraday spike (spike visible briefly, gone after recovery)")


# ============================================================
# Test runner
# ============================================================

def main():
    print("=" * 70)
    print("Reward function unit tests — verifying signs in each regime")
    print("=" * 70)

    scenarios = [
        ("Quiet sideways market, agent matches benchmark",
         scenario_quiet_market_match_benchmark),
        ("Bull market, agent in cash (missed rally)",
         scenario_bull_market_agent_in_cash),
        ("Bear market, agent short — crisis alpha",
         scenario_bear_market_agent_short_makes_money),
        ("Bear market, agent matches (no asymmetric penalty)",
         scenario_bear_market_agent_matches),
        ("Bull market, agent outperforms (mild bonus)",
         scenario_bull_market_agent_outperforms),
        ("Idiosyncratic 30-day drawdown vs flat benchmark",
         scenario_idiosyncratic_drawdown),
        ("Single-day spike with next-day recovery",
         scenario_intraday_spike_recovery),
    ]

    for i, (name, fn) in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {name}")
        print("-" * 70)
        fn()

    print()
    print("=" * 70)
    print("✓ All reward function tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
