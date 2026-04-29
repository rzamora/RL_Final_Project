"""
env/portfolio_hrl_env_joint.py

Joint HL+LL training env. Accepts a 6-dim concatenated action where
  action[0:2] = [gross_signal, net_signal]  (HL action)
  action[2:6] = [w_NVDA_signal, w_AMD_signal, w_SMH_signal, w_TLT_signal]  (LL signal)

Internally splits and applies via parse_hl_action + parse_ll_action.

Obs space: 313 features + 10 portfolio_state + 2 hl_action_slot = 325-dim.
The hl_action_slot in obs gets filled with the HL action that was just used,
mirroring the LL's training-time obs structure for diagnostic compatibility.
This means: at deployment, the obs at step t reflects the HL action chosen
at step t-1 (or zeros at step 0). For joint training during a single env step,
the obs returned AFTER step contains the HL component of the action just taken.

Why: this lets us run the per-HL-action diagnostic on the joint policy
(by manipulating the obs append) the same way we did for v1/v2/v3 LLs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.portfolio_hrl_env_fixed import LowLevelPortfolioEnv


class JointHLLLPortfolioEnv(LowLevelPortfolioEnv):
    """Env for joint HL+LL training.

    Action space: 6-dim Box([-1, +1]).
      Dims 0-1 are the HL action (gross_signal, net_signal).
      Dims 2-5 are the LL action (raw signals projected via parse_ll_action).

    Obs space: same 325-dim as LL env. The trailing 2 dims (hl_action_slot)
    contain the HL action that was just taken — mirroring the LL env's
    structure so that diagnostic code (per-HL-action eval) works unchanged.

    The current LowLevelPortfolioEnv stores fixed_hl_action and uses it on
    every step. We override step() to ignore fixed_hl_action and instead
    use the HL component of the joint action.
    """

    def __init__(self, core):
        # Initialize parent with placeholder fixed_hl_action.
        # We won't use self.fixed_hl_action during step() — instead the joint
        # action's first 2 dims supply the HL action. But _get_obs() in the
        # parent class appends self.fixed_hl_action, so we keep it as a
        # buffer that we overwrite each step with the HL action just used.
        super().__init__(core, fixed_hl_action=np.array([0.0, 0.0], dtype=np.float32))

        # Override action space: 6-dim instead of 4-dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def step(self, joint_action):
        joint_action = np.asarray(joint_action, dtype=np.float32)
        if joint_action.shape != (6,):
            raise ValueError(f"Expected 6-dim joint action, got shape {joint_action.shape}")

        hl_action = joint_action[:2]
        ll_action = joint_action[2:6]

        # Update fixed_hl_action so the next obs's appended HL slot reflects
        # the action just taken (mirrors LL training obs structure)
        self.fixed_hl_action = hl_action.copy()

        target_gross, target_net = self.core.parse_hl_action(hl_action)
        new_weights = self.core.parse_ll_action(ll_action, target_gross, target_net)
        reward, done, info = self.core.apply_allocation(new_weights)
        info["hl_action"] = hl_action.copy()
        info["ll_action"] = ll_action.copy()
        return self._get_obs(), reward, done, False, info