"""
Device benchmark: time 10k PPO steps on CPU vs MPS.

For our small MLP (~250k params), CPU is often faster on Apple Silicon
because the MPS dispatch overhead exceeds the actual compute time per
forward pass. Measure to be sure.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "env"))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from project_config import PATHS
from portfolio_hrl_env_fixed import (
    PortfolioCore, CoreConfig, LowLevelPortfolioEnv,
    LayerNormActorCriticPolicy, process_raw_df,
)


def benchmark(device: str, n_steps: int = 10_000) -> float:
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
        device=device,
        verbose=0,
        seed=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=n_steps)
    elapsed = time.time() - t0
    vec.close()
    return elapsed


def main():
    print("=" * 70)
    print("Step 4a — Device benchmark (CPU vs MPS)")
    print("=" * 70)

    print(f"\nMPS available: {torch.backends.mps.is_available()}")
    print(f"Will time 10,000 PPO steps on each device.\n")

    print("Running on CPU...")
    t_cpu = benchmark("cpu", n_steps=10_000)
    print(f"  CPU: {t_cpu:.1f}s  ({10000/t_cpu:.0f} steps/s)\n")

    if torch.backends.mps.is_available():
        print("Running on MPS...")
        t_mps = benchmark("mps", n_steps=10_000)
        print(f"  MPS: {t_mps:.1f}s  ({10000/t_mps:.0f} steps/s)")
        print()
        if t_cpu < t_mps:
            print(f"→ CPU wins by {(t_mps/t_cpu - 1)*100:.0f}%. Use device='cpu'.")
        else:
            print(f"→ MPS wins by {(t_cpu/t_mps - 1)*100:.0f}%. Use device='mps'.")
    else:
        print("MPS not available — using CPU.")


if __name__ == "__main__":
    main()
