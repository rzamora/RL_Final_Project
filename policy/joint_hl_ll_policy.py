"""
policy/joint_hl_ll_policy.py

Custom ActorCriticPolicy for joint HL+LL training (Option B+B1).

Architecture:
  - Shared LayerNorm MLP trunk (same as LayerNormMlpExtractor)
  - HL head: latent_pi -> 2-dim Diagonal Gaussian (gross_signal, net_signal)
  - LL head: [latent_pi, HL_sample] -> 4-dim Diagonal Gaussian (asset signals)
  - Value head: latent_vf -> scalar V(s)

Forward pass:
  1. features = trunk(obs)
  2. HL_dist = HLHead(features)
  3. HL_action ~ HL_dist.sample()  (or mode if deterministic)
  4. LL_dist = LLHead([features, HL_action])
  5. LL_action ~ LL_dist.sample()
  6. joint_action = concat([HL_action, LL_action])  (6-dim)
  7. log_prob = HL_dist.log_prob(HL_action) + LL_dist.log_prob(LL_action)
  8. entropy = HL_dist.entropy() + LL_dist.entropy()
  9. value = ValueHead(features)

This is Option B1 — LL conditions on the HL's *sampled* action. Gradient
flows through the HL sample via the reparameterization trick (Diagonal
Gaussian's rsample).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
)


class JointDiagGaussianDistribution(Distribution):
    """A 6-dim diagonal Gaussian whose first 2 dims are the HL component
    and last 4 dims are the LL component. log_prob and entropy are sums
    over both components, exposed externally as a single scalar.

    This makes SB3's PPO treat the joint action as a single distribution
    even though internally we computed it as a structured product.
    """

    def __init__(self, hl_dim: int = 2, ll_dim: int = 4):
        super().__init__()
        self.hl_dim = hl_dim
        self.ll_dim = ll_dim
        self.action_dim = hl_dim + ll_dim
        self.hl_dist: Optional[Independent] = None
        self.ll_dist: Optional[Independent] = None
        self.hl_sample: Optional[torch.Tensor] = None

    def proba_distribution_net(self, latent_dim: int):
        # Not used — distribution params are produced inside the policy
        raise NotImplementedError("JointDiagGaussianDistribution params are "
                                  "computed inside JointHLLLPolicy.forward()")

    def proba_distribution(
        self,
        hl_mean: torch.Tensor,
        hl_log_std: torch.Tensor,
        ll_mean: torch.Tensor,
        ll_log_std: torch.Tensor,
        hl_sample: Optional[torch.Tensor] = None,
    ) -> "JointDiagGaussianDistribution":
        """Set the distribution parameters. hl_sample is required when the
        LL distribution depends on the HL sample (it does in our setup)."""
        self.hl_dist = Independent(Normal(hl_mean, hl_log_std.exp()), 1)
        self.ll_dist = Independent(Normal(ll_mean, ll_log_std.exp()), 1)
        self.hl_sample = hl_sample
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Sum of log_probs over HL and LL components."""
        hl_act = actions[..., :self.hl_dim]
        ll_act = actions[..., self.hl_dim:]
        return self.hl_dist.log_prob(hl_act) + self.ll_dist.log_prob(ll_act)

    def entropy(self) -> torch.Tensor:
        """Sum of entropies."""
        return self.hl_dist.entropy() + self.ll_dist.entropy()

    def sample(self) -> torch.Tensor:
        """rsample HL, then sample LL given HL — but this requires re-running
        the LL head with the sampled HL. For the typical PPO use of sample()
        (during rollout), use the policy's forward() instead which computes
        both distributions correctly. This method is here for API compliance.
        """
        hl_sample = self.hl_dist.rsample()
        ll_sample = self.ll_dist.rsample()  # NOTE: assumes ll_dist already conditional on hl_sample
        return torch.cat([hl_sample, ll_sample], dim=-1)

    def mode(self) -> torch.Tensor:
        hl_mode = self.hl_dist.mean
        ll_mode = self.ll_dist.mean
        return torch.cat([hl_mode, ll_mode], dim=-1)

    def actions_from_params(self, *args, deterministic: bool = False, **kwargs):
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(self, *args, **kwargs):
        raise NotImplementedError


class _LayerNormBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class JointHLLLPolicy(ActorCriticPolicy):
    """Joint HL+LL policy with shared trunk and structured heads."""

    HL_DIM = 2
    LL_DIM = 4

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Force the action distribution to be a regular DiagGaussian over the
        # full 6-dim action space; we override the relevant methods to use
        # our structured computation.
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        # We override mlp_extractor and the heads in _build below

    def _build_mlp_extractor(self) -> None:
        # Replace SB3's default mlp_extractor with our LayerNorm trunk.
        # Trunk produces shared latent_pi and latent_vf (here we use the same).
        class _Trunk(nn.Module):
            latent_dim_pi = 256
            latent_dim_vf = 256

            def __init__(self, features_dim: int):
                super().__init__()
                self.policy_net = _LayerNormBlock(features_dim)
                self.value_net = _LayerNormBlock(features_dim)

            def forward(self, features):
                return self.policy_net(features), self.value_net(features)

            def forward_actor(self, features):
                return self.policy_net(features)

            def forward_critic(self, features):
                return self.value_net(features)

        self.mlp_extractor = _Trunk(self.features_dim)

    def _build(self, lr_schedule) -> None:
        """Build the action heads + value head + optimizer.

        Replaces SB3's default action_net (which would be a flat 6-dim head)
        with our structured heads.
        """
        # The standard _build sets self.mlp_extractor (calls _build_mlp_extractor),
        # then creates self.action_net and self.value_net based on self.action_dist.
        # We need to bypass action_net and instead create:
        #   - hl_mean_head, hl_log_std (param)
        #   - ll_mean_head, ll_log_std (param)
        #   - value_head
        #
        # We still call super()._build to get the optimizer, then replace heads.

        super()._build(lr_schedule)

        latent_dim = self.mlp_extractor.latent_dim_pi  # 256

        # HL head: latent -> 2-dim mean
        self.hl_mean_head = nn.Linear(latent_dim, self.HL_DIM)
        # HL log_std as a free parameter (state-independent, like SB3 default)
        self.hl_log_std = nn.Parameter(torch.zeros(self.HL_DIM), requires_grad=True)

        # LL head: [latent + HL_sample] -> 4-dim mean
        self.ll_mean_head = nn.Sequential(
            nn.Linear(latent_dim + self.HL_DIM, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.LL_DIM),
        )
        self.ll_log_std = nn.Parameter(torch.zeros(self.LL_DIM), requires_grad=True)

        # Value head: latent -> scalar
        self.value_net = nn.Linear(latent_dim, 1)

        # We still need an action_net attribute for SB3 internals, but we
        # won't actually use it for forward computation. Set it to a small
        # placeholder so any internal reference doesn't crash.
        self.action_net = nn.Linear(latent_dim, self.HL_DIM + self.LL_DIM)

        # Init weights
        for m in [self.hl_mean_head, self.value_net, self.action_net]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        for m in self.ll_mean_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

        # Re-initialize the optimizer to include the new params
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0), **self.optimizer_kwargs
        )

    def _get_action_dist_from_features(
        self, features: torch.Tensor, deterministic: bool = False
    ) -> Tuple[JointDiagGaussianDistribution, torch.Tensor, torch.Tensor]:
        """Compute structured distribution. Returns (dist, hl_sample, ll_sample).

        If deterministic=True, hl_sample is the HL mean and ll_sample is the
        LL mean (conditioned on HL mean). Otherwise both are reparameterized
        samples.
        """
        latent_pi = self.mlp_extractor.forward_actor(features)

        # HL distribution
        hl_mean = self.hl_mean_head(latent_pi)
        hl_log_std = self.hl_log_std.expand_as(hl_mean)
        hl_dist = Independent(Normal(hl_mean, hl_log_std.exp()), 1)

        # Sample HL action (rsample for gradient flow through B1)
        if deterministic:
            hl_sample = hl_mean
        else:
            hl_sample = hl_dist.rsample()

        # LL distribution conditioned on HL sample
        ll_input = torch.cat([latent_pi, hl_sample], dim=-1)
        ll_mean = self.ll_mean_head(ll_input)
        ll_log_std = self.ll_log_std.expand_as(ll_mean)
        ll_dist = Independent(Normal(ll_mean, ll_log_std.exp()), 1)

        if deterministic:
            ll_sample = ll_mean
        else:
            ll_sample = ll_dist.rsample()

        # Build the joint distribution wrapper
        joint_dist = JointDiagGaussianDistribution(self.HL_DIM, self.LL_DIM)
        joint_dist.proba_distribution(
            hl_mean=hl_mean, hl_log_std=hl_log_std,
            ll_mean=ll_mean, ll_log_std=ll_log_std,
            hl_sample=hl_sample,
        )

        return joint_dist, hl_sample, ll_sample

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns (actions, values, log_prob)."""
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        # Structured action distribution
        joint_dist, hl_sample, ll_sample = self._get_action_dist_from_features(
            features, deterministic=deterministic
        )

        actions = torch.cat([hl_sample, ll_sample], dim=-1)
        log_prob = joint_dist.log_prob(actions)
        values = self.value_net(latent_vf).squeeze(-1)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy of given actions, plus value.

        Critical for PPO update step. We need to recompute the LL distribution
        conditioned on the HL component of the *given* actions (not a freshly
        sampled HL).
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        # HL distribution params
        hl_mean = self.hl_mean_head(latent_pi)
        hl_log_std = self.hl_log_std.expand_as(hl_mean)
        hl_dist = Independent(Normal(hl_mean, hl_log_std.exp()), 1)

        # The HL action that was actually taken (from rollout buffer)
        hl_action_taken = actions[..., :self.HL_DIM]

        # LL distribution conditioned on the HL action that was taken
        ll_input = torch.cat([latent_pi, hl_action_taken], dim=-1)
        ll_mean = self.ll_mean_head(ll_input)
        ll_log_std = self.ll_log_std.expand_as(ll_mean)
        ll_dist = Independent(Normal(ll_mean, ll_log_std.exp()), 1)

        # The LL action that was actually taken
        ll_action_taken = actions[..., self.HL_DIM:]

        # Combined log_prob
        log_prob = hl_dist.log_prob(hl_action_taken) + ll_dist.log_prob(ll_action_taken)

        # Combined entropy
        entropy = hl_dist.entropy() + ll_dist.entropy()

        values = self.value_net(latent_vf).squeeze(-1)

        return values, log_prob, entropy

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Used by .predict() for inference."""
        features = self.extract_features(obs)
        _, hl_sample, ll_sample = self._get_action_dist_from_features(
            features, deterministic=deterministic
        )
        return torch.cat([hl_sample, ll_sample], dim=-1)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf).squeeze(-1)