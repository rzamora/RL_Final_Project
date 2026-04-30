"""
env/portfolio_hrl_env_no_regime.py

Variant of portfolio_hrl_env_fixed.py that strips the four regime_prob_*
columns from the feature space. Both LL and HL operate on the reduced
309-dim feature set + portfolio_state.

Hypothesis: the regime classifier degrades on test (sign flips on equity
returns for Bull and SevereBear regimes). Removing its outputs from the
observation should let the policy condition on transferable features only.

Differences from portfolio_hrl_env_fixed.py:
  - process_raw_df also drops regime_prob_* columns
  - load_synthetic_pool also strips regime_prob_* by feature_names
  - LowLevelPortfolioEnv obs_dim = 309 + 10 + 2 = 321
  - HighLevelPortfolioEnv obs_dim = 309 + 10 = 319
  - Reward function unchanged
  - PPO policy class unchanged

Run order:
  1. ll_random_hl_synth_pretrain_no_regime.py     (~10 min)
  2. ll_random_hl_finetune_no_regime.py           (~3 min)
  3. hl_synth_pretrain_no_regime.py               (~10 min)
  4. hl_finetune_real_no_regime.py                (~3 min)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from torch import nn


# ============================================================
# Schema
# ============================================================

PRICE_COLS = ("NVDA_close", "AMD_close", "SMH_close", "TLT_close")
REGIME_PROB_COLS = (
    "regime_prob_Bull",
    "regime_prob_Bear",
    "regime_prob_SevereBear",
    "regime_prob_Crisis",
)


# ============================================================
# Data loading — strips regime_prob_* in addition to *_close
# ============================================================

def process_raw_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert merged CSV to (features, returns, prices), dropping both
    *_close columns AND regime_prob_* columns from the feature array.
    """
    price_cols = list(PRICE_COLS)
    regime_cols = [c for c in REGIME_PROB_COLS if c in df.columns]

    # Drop date, prices, and regime probs from features
    drop_cols = ["date"] + price_cols + regime_cols
    feats_df = df.drop(columns=drop_cols)
    features = feats_df.to_numpy(dtype=np.float32)
    prices = df[price_cols].to_numpy(dtype=np.float32)

    rets_df = df[price_cols].pct_change()
    rets_df = rets_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns = rets_df.to_numpy(dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, returns, prices


def load_synthetic_pool(npz_path: str, drop_close_features: bool = True) -> dict:
    """Load synthetic pool, stripping both *_close AND regime_prob_* features.

    Returns rescale unchanged: synth pool stores returns in percent, real CSVs
    use fractional. We rescale synth returns by /100 so both sources reach the
    env in fractional units.
    """
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    returns = data["returns"].astype(np.float32) / 100.0
    feature_names = list(data["feature_names"]) if "feature_names" in data.files else None

    if feature_names is None:
        raise ValueError(
            "Pool has no feature_names. Cannot reliably strip regime_prob columns. "
            "Use the original pool with feature_names embedded."
        )

    # Build set of columns to drop: prices (if drop_close_features) + regime probs
    drop_cols = list(REGIME_PROB_COLS)
    if drop_close_features:
        drop_cols += list(PRICE_COLS)

    drop_indices = [i for i, n in enumerate(feature_names) if n in drop_cols]
    if not drop_indices:
        raise ValueError(f"No columns matching {drop_cols} found in pool")

    # Extract prices BEFORE dropping (for benchmark computation)
    prices = None
    if drop_close_features:
        close_idx = [i for i, n in enumerate(feature_names) if n in PRICE_COLS]
        if len(close_idx) != len(PRICE_COLS):
            raise ValueError(
                f"Expected {len(PRICE_COLS)} close columns, found {len(close_idx)}"
            )
        name_to_idx = {feature_names[i]: i for i in close_idx}
        ordered_idx = [name_to_idx[name] for name in PRICE_COLS]
        prices = features[:, :, ordered_idx].astype(np.float32)

    # Drop both close and regime_prob columns from features
    keep_mask = np.ones(features.shape[2], dtype=bool)
    for i in drop_indices:
        keep_mask[i] = False
    features = features[:, :, keep_mask]
    feature_names = [n for i, n in enumerate(feature_names) if keep_mask[i]]

    return {
        "features": features,
        "returns": returns,
        "prices": prices,
        "regimes": data["regimes"] if "regimes" in data.files else None,
        "feature_names": feature_names,
    }


# ============================================================
# Config — unchanged
# ============================================================

@dataclass
class CoreConfig:
    max_gross: float = 1.5
    initial_equity: float = 1.0
    episode_length: int = 384

    dd_window_short: int = 21
    dd_window_long: int = 63
    dd_threshold_short: float = 0.05
    dd_threshold_long: float = 0.10
    lambda_dd_short: float = 10.0
    lambda_dd_long: float = 10.0

    lambda_turnover: float = 0.01

    benchmark_window: int = 63
    benchmark_warmup: int = 21
    benchmark_threshold: float = 0.02

    lambda_upside_miss: float = 0.5
    lambda_upside_beat: float = 0.1
    lambda_downside_excess: float = 0.1
    lambda_crisis_alpha: float = 0.5


# ============================================================
# Portfolio core — unchanged from original
# ============================================================

class PortfolioCore:
    """Stateful portfolio simulator. Identical to the original — the
    feature_dim is just smaller now (309 instead of 313)."""

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        cfg: Optional[CoreConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)

        if benchmark_returns is None:
            benchmark_returns = self.returns[:, :3].mean(axis=1)
        self.benchmark_returns = benchmark_returns.astype(np.float32)

        self.cfg = cfg or CoreConfig()
        self.rng = rng or np.random.default_rng()

        self.full_n_steps = len(self.features)
        self.feature_dim = self.features.shape[1]
        self.n_assets = self.returns.shape[1]

        if self.full_n_steps < self.cfg.episode_length + 1:
            raise ValueError(
                f"Data too short: {self.full_n_steps} rows < "
                f"{self.cfg.episode_length} episode length"
            )

        self.reset()

    def _init_windows(self):
        self.equity_window_short      = deque(maxlen=self.cfg.dd_window_short)
        self.equity_window_long       = deque(maxlen=self.cfg.dd_window_long)
        self.bench_equity_window_short = deque(maxlen=self.cfg.dd_window_short)
        self.bench_equity_window_long  = deque(maxlen=self.cfg.dd_window_long)
        self.excess_window            = deque(maxlen=self.cfg.benchmark_window)
        self.bench_window             = deque(maxlen=self.cfg.benchmark_window)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_start = self.full_n_steps - self.cfg.episode_length - 1
        self.t_start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self.t_end = self.t_start + self.cfg.episode_length
        self.t = self.t_start

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

    @property
    def steps_remaining(self) -> int:
        return self.t_end - self.t

    def portfolio_state(self) -> np.ndarray:
        gross = float(np.sum(np.abs(self.weights)))
        net = float(np.sum(self.weights))
        return np.array(
            [
                self.equity,
                self.excess_dd_short,
                gross,
                net,
                *self.weights,
                self.quarterly_excess,
                self.quarterly_bench,
            ],
            dtype=np.float32,
        )

    def obs(self) -> np.ndarray:
        return np.concatenate([self.features[self.t], self.portfolio_state()]).astype(np.float32)

    def parse_hl_action(self, hl_action: np.ndarray) -> Tuple[float, float]:
        gross_raw, net_raw = float(hl_action[0]), float(hl_action[1])
        target_gross = (gross_raw + 1.0) / 2.0 * self.cfg.max_gross
        target_net = net_raw * target_gross
        return target_gross, target_net

    def parse_ll_action(
        self,
        ll_action: np.ndarray,
        target_gross: float,
        target_net: float,
    ) -> np.ndarray:
        raw = np.asarray(ll_action, dtype=np.float64)
        target_gross = float(np.clip(target_gross, 0.0, self.cfg.max_gross))
        target_net = float(np.clip(target_net, -target_gross, target_gross))

        long_gross = 0.5 * (target_gross + target_net)
        short_gross = 0.5 * (target_gross - target_net)

        pos = np.maximum(raw, 0.0)
        neg = np.maximum(-raw, 0.0)
        pos_sum = pos.sum()
        neg_sum = neg.sum()

        if pos_sum > 1e-8:
            long_book = (pos / pos_sum) * long_gross
        elif long_gross > 1e-8:
            long_book = np.full(self.n_assets, long_gross / self.n_assets)
        else:
            long_book = np.zeros(self.n_assets)

        if neg_sum > 1e-8:
            short_book = (neg / neg_sum) * short_gross
        elif short_gross > 1e-8:
            short_book = np.full(self.n_assets, short_gross / self.n_assets)
        else:
            short_book = np.zeros(self.n_assets)

        return (long_book - short_book).astype(np.float32)

    def apply_allocation(self, new_weights: np.ndarray) -> Tuple[float, bool, dict]:
        old_weights = self.weights.copy()
        self.weights = new_weights.astype(np.float32)

        next_t = self.t + 1
        if next_t >= len(self.returns):
            info = {
                "equity": self.equity,
                "bench_equity": self.bench_equity,
                "excess_dd_short": self.excess_dd_short,
                "excess_dd_long": self.excess_dd_long,
                "agent_dd_short": 0.0,
                "bench_dd_short": 0.0,
                "weights": self.weights.copy(),
                "turnover": float(np.sum(np.abs(self.weights - old_weights))),
                "portfolio_return": 0.0,
                "benchmark_return": 0.0,
                "quarterly_excess": self.quarterly_excess,
                "quarterly_bench": self.quarterly_bench,
            }
            self.t = next_t
            return 0.0, True, info

        asset_returns = self.returns[next_t]
        portfolio_return = float(np.dot(self.weights, asset_returns))
        bench_return = float(self.benchmark_returns[next_t])

        self.equity *= 1.0 + portfolio_return
        self.bench_equity *= 1.0 + bench_return

        self.equity_window_short.append(self.equity)
        self.equity_window_long.append(self.equity)
        self.bench_equity_window_short.append(self.bench_equity)
        self.bench_equity_window_long.append(self.bench_equity)

        agent_peak_short = max(self.equity_window_short)
        bench_peak_short = max(self.bench_equity_window_short)
        agent_dd_short = (self.equity / agent_peak_short - 1.0) if agent_peak_short > 0 else 0.0
        bench_dd_short = (self.bench_equity / bench_peak_short - 1.0) if bench_peak_short > 0 else 0.0
        self.excess_dd_short = float(agent_dd_short - bench_dd_short)

        agent_peak_long = max(self.equity_window_long)
        bench_peak_long = max(self.bench_equity_window_long)
        agent_dd_long = (self.equity / agent_peak_long - 1.0) if agent_peak_long > 0 else 0.0
        bench_dd_long = (self.bench_equity / bench_peak_long - 1.0) if bench_peak_long > 0 else 0.0
        self.excess_dd_long = float(agent_dd_long - bench_dd_long)

        excess_return = portfolio_return - bench_return
        self.excess_window.append(excess_return)
        self.bench_window.append(bench_return)

        if len(self.excess_window) >= self.cfg.benchmark_warmup:
            self.quarterly_excess = float(sum(self.excess_window))
            self.quarterly_bench = float(sum(self.bench_window))
        else:
            self.quarterly_excess = 0.0
            self.quarterly_bench = 0.0

        turnover = float(np.sum(np.abs(self.weights - old_weights)))

        info = {
            "equity": self.equity,
            "bench_equity": self.bench_equity,
            "excess_dd_short": self.excess_dd_short,
            "excess_dd_long": self.excess_dd_long,
            "agent_dd_short": float(agent_dd_short),
            "bench_dd_short": float(bench_dd_short),
            "weights": self.weights.copy(),
            "turnover": turnover,
            "portfolio_return": portfolio_return,
            "benchmark_return": bench_return,
            "quarterly_excess": self.quarterly_excess,
            "quarterly_bench": self.quarterly_bench,
        }

        self.t = next_t
        reward = self._compute_reward(turnover)
        done = self.t >= self.t_end
        return reward, done, info

    def _compute_reward(self, turnover: float) -> float:
        reward = 0.0

        if abs(self.excess_dd_short) > self.cfg.dd_threshold_short:
            excess_above_threshold = abs(self.excess_dd_short) - self.cfg.dd_threshold_short
            reward -= self.cfg.lambda_dd_short * excess_above_threshold

        if abs(self.excess_dd_long) > self.cfg.dd_threshold_long:
            excess_above_threshold = abs(self.excess_dd_long) - self.cfg.dd_threshold_long
            reward -= self.cfg.lambda_dd_long * excess_above_threshold

        reward -= self.cfg.lambda_turnover * turnover

        if len(self.excess_window) >= self.cfg.benchmark_warmup:
            excess_above_band = self.quarterly_excess - self.cfg.benchmark_threshold
            excess_below_band = -self.cfg.benchmark_threshold - self.quarterly_excess

            if self.quarterly_bench >= 0:
                if excess_below_band > 0:
                    reward -= self.cfg.lambda_upside_miss * excess_below_band
                elif excess_above_band > 0:
                    reward += self.cfg.lambda_upside_beat * excess_above_band
            else:
                if excess_below_band > 0:
                    reward -= self.cfg.lambda_downside_excess * excess_below_band
                elif excess_above_band > 0:
                    reward += self.cfg.lambda_crisis_alpha * excess_above_band

        return float(reward)


# ============================================================
# Pool sampler
# ============================================================

class SyntheticPoolCoreSampler:
    def __init__(
        self,
        pool: dict,
        cfg: Optional[CoreConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.pool_features = pool["features"]
        self.pool_returns = pool["returns"]
        self.cfg = cfg or CoreConfig()
        self.rng = rng or np.random.default_rng()

        self.n_paths, self.path_T, self.feature_dim = self.pool_features.shape
        self.n_assets = self.pool_returns.shape[2]

        if self.cfg.episode_length > self.path_T:
            raise ValueError(
                f"episode_length {self.cfg.episode_length} > pool path T {self.path_T}"
            )

        self.current_path = None
        self._core: Optional[PortfolioCore] = None
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_path = int(self.rng.integers(0, self.n_paths))
        feats = self.pool_features[self.current_path]
        rets = self.pool_returns[self.current_path]
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        self._core = PortfolioCore(feats, rets, cfg=self.cfg, rng=self.rng)

    def __getattr__(self, name):
        return getattr(self._core, name)


# ============================================================
# Envs — same as original but with smaller obs_dim
# ============================================================

class LowLevelPortfolioEnv(gym.Env):
    """LL env. obs = features (309) + portfolio_state (10) + hl_action (2) = 321."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        core,
        fixed_hl_action: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.core = core
        self.fixed_hl_action = (
            np.array([0.33, 0.5], dtype=np.float32)
            if fixed_hl_action is None
            else np.asarray(fixed_hl_action, dtype=np.float32)
        )

        portfolio_state_dim = 4 + core.n_assets + 2
        obs_dim = core.feature_dim + portfolio_state_dim + 2
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(core.n_assets,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset(seed=seed)
        return self._get_obs(), {}

    def step(self, ll_action):
        target_gross, target_net = self.core.parse_hl_action(self.fixed_hl_action)
        new_weights = self.core.parse_ll_action(ll_action, target_gross, target_net)
        reward, done, info = self.core.apply_allocation(new_weights)
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        return np.concatenate([self.core.obs(), self.fixed_hl_action]).astype(np.float32)


class LowLevelPortfolioEnvRandomHL(LowLevelPortfolioEnv):
    """LL training env where HL action is sampled uniformly per episode.
    Forces the LL to learn responses across the full HL action space."""

    def __init__(self, core, hl_seed: int = 7919):
        super().__init__(core, fixed_hl_action=None)
        self.hl_rng = np.random.default_rng(hl_seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.hl_rng = np.random.default_rng(seed + 7919)
        new_hl = self.hl_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.fixed_hl_action = new_hl
        return super().reset(seed=seed, options=options)


class HighLevelPortfolioEnv(gym.Env):
    """HL env. obs = features (309) + portfolio_state (10) = 319."""

    metadata = {"render_modes": []}

    def __init__(self, core, ll_model: PPO):
        super().__init__()
        self.core = core
        self.ll_model = ll_model

        portfolio_state_dim = 4 + core.n_assets + 2
        obs_dim = core.feature_dim + portfolio_state_dim
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset(seed=seed)
        return self.core.obs(), {}

    def step(self, hl_action):
        target_gross, target_net = self.core.parse_hl_action(hl_action)
        ll_obs = np.concatenate([self.core.obs(), hl_action]).astype(np.float32)
        ll_action, _ = self.ll_model.predict(ll_obs, deterministic=False)
        new_weights = self.core.parse_ll_action(ll_action, target_gross, target_net)
        reward, done, info = self.core.apply_allocation(new_weights)
        info["hl_action"] = np.asarray(hl_action)
        info["ll_action"] = np.asarray(ll_action)
        return self.core.obs(), reward, done, False, info


# ============================================================
# Policy — unchanged
# ============================================================

class LayerNormMlpExtractor(nn.Module):
    def __init__(self, features_dim: int):
        super().__init__()

        def block():
            return nn.Sequential(
                nn.Linear(features_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
            )

        self.policy_net = block()
        self.value_net = block()
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class LayerNormActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = LayerNormMlpExtractor(self.features_dim)