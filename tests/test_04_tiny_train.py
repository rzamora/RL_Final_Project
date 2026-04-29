"""
Step 4 — Tiny PPO training run.

Train PPO for 50,000 steps on real data, watching for ep_rew_mean to trend
upward. Uses 2 parallel envs, CPU device, VecNormalize on observations.

Goal: verify the env is *learnable*. Don't aim for great performance — just
upward reward trend.

Run from project root: `python tests/test_04_tiny_train.py`
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "env"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from project_config import PATHS
from portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig, LowLevelPortfolioEnv,
    LayerNormActorCriticPolicy, process_raw_df,
)


class RewardTrendCallback(BaseCallback):
    """Records ep_rew_mean at each rollout completion."""
    def __init__(self):
        super().__init__()
        self.history = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            self.history.append({
                "timesteps": self.num_timesteps,
                "ep_rew_mean": float(np.mean(ep_rewards)),
                "ep_len_mean": float(np.mean([ep["l"] for ep in self.model.ep_info_buffer])),
            })


def main():
    print("=" * 70)
    print("Step 4 — Tiny PPO training run (50k steps)")
    print("=" * 70)

    train_df = pd.read_csv(PATHS.train_csv)
    feats, rets, _ = process_raw_df(train_df)
    cfg = CoreConfig(episode_length=384)

    def make_env(seed):
        def _init():
            core = PortfolioCore(feats, rets, cfg=cfg,
                                  rng=np.random.default_rng(seed))
            return LowLevelPortfolioEnv(core)
        return _init

    vec = DummyVecEnv([make_env(i) for i in range(2)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        LayerNormActorCriticPolicy,
        vec,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
        verbose=1,
        seed=0,
        tensorboard_log=str(PATHS.tb_logs),
    )

    callback = RewardTrendCallback()

    print(f"\nStarting 50,000 step training run...")
    print(f"  Watch for ep_rew_mean trending upward.")
    print(f"  Tensorboard logs: {PATHS.tb_logs}")
    print()

    model.learn(total_timesteps=50_000, callback=callback)

    PATHS.checkpoints.mkdir(parents=True, exist_ok=True)
    model.save(PATHS.checkpoints / "tiny_train_50k")
    vec.save(str(PATHS.checkpoints / "tiny_train_50k_vecnorm.pkl"))

    # ---------- Reward trend diagnostic ----------
    print()
    print("=" * 70)
    print("Reward trend diagnostic")
    print("=" * 70)
    print(f"\n{'Timesteps':>10s}  {'ep_rew_mean':>14s}  {'ep_len_mean':>12s}")
    print("-" * 42)
    for entry in callback.history:
        print(f"{entry['timesteps']:>10d}  {entry['ep_rew_mean']:>+14.3f}  {entry['ep_len_mean']:>12.0f}")

    # First-vs-last quartile comparison
    if len(callback.history) >= 8:
        n_q = len(callback.history) // 4
        first_q = np.mean([e["ep_rew_mean"] for e in callback.history[:n_q]])
        last_q = np.mean([e["ep_rew_mean"] for e in callback.history[-n_q:]])
        print()
        print(f"First quartile mean:  {first_q:+.3f}")
        print(f"Last quartile mean:   {last_q:+.3f}")
        print(f"Improvement:          {last_q - first_q:+.3f}")

        if last_q > first_q + 5.0:
            verdict = "✓ STRONG upward trend — learning"
        elif last_q > first_q + 1.0:
            verdict = "✓ mild upward trend — learning slowly"
        elif last_q > first_q - 1.0:
            verdict = "○ flat — needs more steps or env tweak"
        else:
            verdict = "✗ DOWNWARD trend — something broken"

        print(f"Verdict: {verdict}")

    print()
    print("=" * 70)
    print("Step 4 complete. Saved model and vecnorm to checkpoints/")
    print("=" * 70)


if __name__ == "__main__":
    main()
