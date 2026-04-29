"""
env/portfolio_hrl_env_random_hl.py

LL training environment that samples HL actions uniformly per episode.

The original LowLevelPortfolioEnv fixes the HL action at construction time
(default [0.33, 0.5]). This means the LL never sees other HL action values
during training, and its VecNormalize stats reflect that. When such an LL is
later wrapped by HighLevelPortfolioEnv (where the HL produces varying actions),
the LL's response to off-distribution HL actions is unpredictable, which
breaks HL training (see HL frozen-LL pretrain results in the project report).

Fix: this env class samples [gross_signal, net_signal] ~ Uniform(-1, +1)^2
on every reset(). The LL learns to produce sensible weights given any HL
action it might encounter when later combined with a trained HL.

Usage matches LowLevelPortfolioEnv otherwise — same obs space (325-dim),
same action space (4-dim), same reward.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.portfolio_hrl_env_fixed import LowLevelPortfolioEnv


class LowLevelPortfolioEnvRandomHL(LowLevelPortfolioEnv):
    """LL env with HL action randomized per episode.

    Key differences from base class:
      - HL action is freshly sampled from Uniform([-1,1]^2) on every reset()
      - The sampled HL action is held constant for the full episode and
        appended to obs each step (same as base class behavior with its
        fixed_hl_action attribute)
      - Adds a separate np.random.Generator for HL action sampling so seeding
        is reproducible without interfering with the core's RNG
    """

    def __init__(
        self,
        core,
        hl_rng: Optional[np.random.Generator] = None,
    ):
        # Call base init with a placeholder fixed_hl_action — we'll overwrite
        # self.fixed_hl_action in reset()
        super().__init__(core, fixed_hl_action=np.array([0.0, 0.0], dtype=np.float32))
        self.hl_rng = hl_rng or np.random.default_rng()

    def reset(self, seed=None, options=None):
        # If a seed is provided, also reseed the HL RNG so episodes are
        # reproducible end-to-end
        if seed is not None:
            self.hl_rng = np.random.default_rng(seed + 7919)  # offset to avoid identical sampling
        # Sample new HL action for this episode
        new_hl = self.hl_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.fixed_hl_action = new_hl
        # Delegate the rest of reset to base class
        return super().reset(seed=seed, options=options)