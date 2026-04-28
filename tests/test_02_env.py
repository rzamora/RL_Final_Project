"""
Step 2 — Environment smoke test (updated for asymmetric reward function).

Verifies that PortfolioCore + LowLevelPortfolioEnv:
  - Construct without errors
  - Run a complete episode of exactly cfg.episode_length steps
  - Produce observations of the expected shape (325 dim for LL env)
  - Keep equity positive throughout
  - Keep excess_dd_short within [-1, 1]
  - Honor the gross leverage cap (|w|.sum() <= max_gross)
  - Populate the new info dict correctly

Run from project root: `python tests/test_02_env.py`
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "env"))

import numpy as np
import pandas as pd
from project_config import PATHS
from portfolio_hrl_env_fixed import (
    PortfolioCore,
    CoreConfig,
    LowLevelPortfolioEnv,
    process_raw_df,
)


def main():
    print("=" * 70)
    print("Step 2 — Environment smoke test")
    print("=" * 70)

    # ---------- Load real data ----------
    train_df = pd.read_csv(PATHS.train_csv)
    feats, rets, prices = process_raw_df(train_df)
    print(f"Loaded {len(feats)} days of training data")
    print(f"  Features: {feats.shape}, Returns: {rets.shape}")

    # ---------- Construct core + env ----------
    cfg = CoreConfig(episode_length=384)
    core = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(42))
    env = LowLevelPortfolioEnv(core)

    print(f"\nCore configured:")
    print(f"  episode_length:    {core.cfg.episode_length}")
    print(f"  max_gross:         {core.cfg.max_gross}")
    print(f"  dd_window_short:   {core.cfg.dd_window_short}d, threshold {core.cfg.dd_threshold_short:.0%}")
    print(f"  dd_window_long:    {core.cfg.dd_window_long}d, threshold {core.cfg.dd_threshold_long:.0%}")
    print(f"  benchmark_window:  {core.cfg.benchmark_window}d, threshold {core.cfg.benchmark_threshold:.0%}")
    print(f"  lambda upside:     miss={core.cfg.lambda_upside_miss}, beat={core.cfg.lambda_upside_beat}")
    print(f"  lambda downside:   crisis_alpha={core.cfg.lambda_crisis_alpha}, excess={core.cfg.lambda_downside_excess}")
    print(f"  feature_dim:       {core.feature_dim}")
    print(f"  n_assets:          {core.n_assets}")

    # ---------- Verify observation/action spaces ----------
    # Portfolio state has 10 elements:
    #   equity, excess_dd_short, gross, net, w[4], q_excess, q_bench
    # LL obs = 313 (features) + 10 (state) + 2 (HL action) = 325
    portfolio_state_dim = 4 + core.n_assets + 2  # 4 scalars + n_assets + 2 quarterly
    obs_dim_expected = core.feature_dim + portfolio_state_dim + 2
    print(f"\nObs space: {env.observation_space.shape}  (expected: ({obs_dim_expected},))")
    assert env.observation_space.shape == (obs_dim_expected,), (
        f"Obs dim mismatch: got {env.observation_space.shape}, "
        f"expected ({obs_dim_expected},)"
    )
    print(f"Action space: {env.action_space.shape}  (expected: ({core.n_assets},))")
    assert env.action_space.shape == (core.n_assets,)

    # ---------- Run one full episode ----------
    print(f"\n--- Running episode with random actions ---")
    obs, info = env.reset(seed=42)
    print(f"After reset: obs.shape={obs.shape}, t_start={core.t_start}, t_end={core.t_end}")
    assert obs.shape == (obs_dim_expected,)
    assert obs.dtype == np.float32
    assert not np.any(np.isnan(obs)), "Reset obs has NaN"
    assert not np.any(np.isinf(obs)), "Reset obs has Inf"

    equity_curve = [core.equity]
    bench_curve = [core.bench_equity]
    rewards = []
    grosses = []
    nets = []
    excess_dds_short = []
    quarterly_excess_history = []
    turnovers = []

    step = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        equity_curve.append(info["equity"])
        bench_curve.append(info["bench_equity"])
        excess_dds_short.append(info["excess_dd_short"])
        quarterly_excess_history.append(info["quarterly_excess"])
        turnovers.append(info["turnover"])
        grosses.append(float(np.sum(np.abs(info["weights"]))))
        nets.append(float(np.sum(info["weights"])))
        step += 1

        # Step-level invariants
        assert info["equity"] > 0, f"Equity went non-positive at step {step}: {info['equity']}"
        assert -1.0 <= info["excess_dd_short"] <= 1.0, (
            f"excess_dd_short OOB at step {step}: {info['excess_dd_short']}"
        )
        assert grosses[-1] <= cfg.max_gross + 1e-5, (
            f"Gross leverage breach at step {step}: {grosses[-1]:.4f} > {cfg.max_gross}"
        )
        assert obs.shape == (obs_dim_expected,)

        if terminated or truncated:
            break

    # ---------- Episode-level checks ----------
    print(f"\nEpisode finished: {step} steps")
    assert step == cfg.episode_length, (
        f"Expected {cfg.episode_length} steps, got {step}"
    )

    # ---------- Summary stats ----------
    rewards = np.array(rewards)
    grosses = np.array(grosses)
    nets = np.array(nets)
    excess_dds_short = np.array(excess_dds_short)
    quarterly_excess_history = np.array(quarterly_excess_history)
    turnovers = np.array(turnovers)

    print(f"\n--- Summary across the episode ---")
    print(f"  Reward:           mean={rewards.mean():+.5f}  std={rewards.std():.5f}  "
          f"min={rewards.min():+.5f}  max={rewards.max():+.5f}")
    print(f"  Equity:           start={equity_curve[0]:.4f}  end={equity_curve[-1]:.4f}  "
          f"max={max(equity_curve):.4f}  min={min(equity_curve):.4f}")
    print(f"  Bench equity:     start={bench_curve[0]:.4f}  end={bench_curve[-1]:.4f}")
    print(f"  Gross:            mean={grosses.mean():.3f}  max={grosses.max():.3f}  "
          f"(cap: {cfg.max_gross})")
    print(f"  Net:              mean={nets.mean():+.3f}  min={nets.min():+.3f}  "
          f"max={nets.max():+.3f}")
    print(f"  Excess_dd_short:  mean={excess_dds_short.mean():+.3f}  "
          f"min={excess_dds_short.min():+.3f}  max={excess_dds_short.max():+.3f}")
    print(f"  Quarterly excess: mean={quarterly_excess_history.mean():+.4f}  "
          f"min={quarterly_excess_history.min():+.4f}  max={quarterly_excess_history.max():+.4f}")
    print(f"  Turnover:         mean={turnovers.mean():.4f}  max={turnovers.max():.4f}")

    # ---------- Sanity sniff tests ----------
    print(f"\n--- Sanity sniff tests ---")

    equity_change = abs(equity_curve[-1] / equity_curve[0] - 1.0)
    print(f"  Agent equity moved {equity_change*100:.1f}% from start to end")
    bench_change = abs(bench_curve[-1] / bench_curve[0] - 1.0)
    print(f"  Bench equity moved {bench_change*100:.1f}% from start to end")
    if equity_change < 0.01:
        print(f"  ⚠ Agent ended within 1% of starting equity. Env may be stuck.")

    # Confirm parallel benchmark equity tracking
    print(f"  Final agent: {equity_curve[-1]:.4f}, final bench: {bench_curve[-1]:.4f}")
    print(f"  Net exposure crossed zero: {(np.diff(np.sign(nets)) != 0).any()}")

    # Reset twice and confirm we get a different starting point
    core.reset(seed=0)
    t0 = core.t_start
    core.reset(seed=1)
    t1 = core.t_start
    print(f"  Random episode starts: seed=0 → t={t0}, seed=1 → t={t1}  "
          f"(should differ: {'✓' if t0 != t1 else '✗ same start, RNG broken'})")

    print()
    print("=" * 70)
    print("✓ Step 2 PASSED — env constructs, runs, and respects all invariants")
    print("=" * 70)


if __name__ == "__main__":
    main()
