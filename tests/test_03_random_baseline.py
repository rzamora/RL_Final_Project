"""
Step 3 — Random policy baseline.

Records the distribution of equity, Sharpe, drawdown over 20 random episodes.
These are the numbers any trained agent must beat.

Run from project root: `python tests/test_03_random_baseline.py`
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "env"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import numpy as np
import pandas as pd
from project_config import PATHS
from portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig, LowLevelPortfolioEnv, process_raw_df,
)
from portfolio_stats import compute_stats


def main():
    print("=" * 70)
    print("Step 3 — Random policy baseline")
    print("=" * 70)

    train_df = pd.read_csv(PATHS.train_csv)
    feats, rets, _ = process_raw_df(train_df)
    cfg = CoreConfig(episode_length=384)

    n_episodes = 20
    print(f"\nRunning {n_episodes} random-action episodes...")

    finals, sharpes, max_dds, mean_turnovers, total_rewards = [], [], [], [], []

    for seed in range(n_episodes):
        core = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(seed))
        env = LowLevelPortfolioEnv(core)
        obs, _ = env.reset(seed=seed)

        equity = [core.equity]
        turnovers = []
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            equity.append(info["equity"])
            turnovers.append(info["turnover"])
            ep_reward += reward
            done = terminated or truncated

        equity = np.array(equity)
        stats = compute_stats(equity, label=f"random_{seed}")
        finals.append(equity[-1])
        sharpes.append(stats.sharpe)
        max_dds.append(stats.max_drawdown)
        mean_turnovers.append(np.mean(turnovers))
        total_rewards.append(ep_reward)

    finals = np.array(finals)
    sharpes = np.array(sharpes)
    max_dds = np.array(max_dds)
    mean_turnovers = np.array(mean_turnovers)
    total_rewards = np.array(total_rewards)

    print(f"\n--- Distribution across {n_episodes} random episodes ---")
    print(f"{'Metric':<20s} {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print(f"{'-'*65}")
    print(f"{'Final equity':<20s} {finals.mean():>10.3f}  {finals.std():>10.3f}  "
          f"{finals.min():>10.3f}  {finals.max():>10.3f}")
    print(f"{'Sharpe':<20s} {sharpes.mean():>10.3f}  {sharpes.std():>10.3f}  "
          f"{sharpes.min():>10.3f}  {sharpes.max():>10.3f}")
    print(f"{'Max drawdown':<20s} {max_dds.mean():>10.3f}  {max_dds.std():>10.3f}  "
          f"{max_dds.min():>10.3f}  {max_dds.max():>10.3f}")
    print(f"{'Mean turnover':<20s} {mean_turnovers.mean():>10.3f}  {mean_turnovers.std():>10.3f}  "
          f"{mean_turnovers.min():>10.3f}  {mean_turnovers.max():>10.3f}")
    print(f"{'Total reward':<20s} {total_rewards.mean():>10.3f}  {total_rewards.std():>10.3f}  "
          f"{total_rewards.min():>10.3f}  {total_rewards.max():>10.3f}")

    print(f"\n--- Reference numbers any trained agent must beat ---")
    print(f"  Median final equity: {np.median(finals):.3f}")
    print(f"  Median Sharpe:       {np.median(sharpes):+.3f}")
    print(f"  Median max DD:       {np.median(max_dds):.3f}")
    print(f"  Median total reward: {np.median(total_rewards):+.3f}")

    print()
    print("=" * 70)
    print("✓ Step 3 complete — random baseline recorded")
    print("=" * 70)


if __name__ == "__main__":
    main()
