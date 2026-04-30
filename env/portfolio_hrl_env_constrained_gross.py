"""
Portfolio HRL environment with rolling-window risk and asymmetric benchmark reward.

Reward design philosophy:
- Penalize sustained drawdowns relative to benchmark (rolling 21d and 63d windows),
  not single-day equity dips. A real PM is judged on monthly losses, not intraday vol.
- Reward upside-capture and crisis alpha asymmetrically:
    Up market, missed rally:        HEAVY penalty (career risk)
    Up market, outperformance:      mild bonus
    Down market, excess loss:       mild penalty (acceptable in selloffs)
    Down market, made money/up:     HEAVY bonus (crisis alpha is the prize)
- This produces a policy that participates in rallies, protects in selloffs,
  and is genuinely incentivized to generate alpha during crises.

Changes vs. previous version:
- Drawdown is computed on rolling windows (21d, 63d), not all-time peak
- Drawdown is measured as EXCESS over benchmark (agent_dd - bench_dd), not absolute
- Benchmark tracking is over a 63-day cumulative excess window with 21-day warmup
- Asymmetric reward shape: 4 separate lambdas for the up/down × beat/miss matrix
- Obs vector exposes excess_dd_short, quarterly_excess, quarterly_bench to the agent
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
# Data loading
# ============================================================

PRICE_COLS = ("NVDA_close", "AMD_close", "SMH_close", "TLT_close")


def process_raw_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the merged CSV to (features, returns, prices).

    Drops *_close columns from features (they're absolute price levels that
    leak the calendar position). Prices kept on the side for return calc.
    """
    price_cols = list(PRICE_COLS)
    feats_df = df.drop(columns=["date"] + price_cols)
    features = feats_df.to_numpy(dtype=np.float32)
    prices = df[price_cols].to_numpy(dtype=np.float32)

    rets_df = df[price_cols].pct_change()
    rets_df = rets_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns = rets_df.to_numpy(dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, returns, prices


def load_synthetic_pool(npz_path: str, drop_close_features: bool = True) -> dict:
    """Load synthetic pool, optionally stripping *_close features (recommended).

    NOTE: synth_pool.npz stores returns in percent units (e.g. 1.27 = +1.27%),
    while the real CSVs produce fractional returns via pct_change() (e.g.
    0.0127 = +1.27%). PortfolioCore expects fractional. We rescale here so
    both data sources reach the env in the same units.
    """
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    returns = data["returns"].astype(np.float32) / 100.0
    feature_names = list(data["feature_names"]) if "feature_names" in data.files else None

    prices = None
    if drop_close_features and feature_names is not None:
        close_idx = [i for i, n in enumerate(feature_names) if n in PRICE_COLS]
        if len(close_idx) != len(PRICE_COLS):
            raise ValueError(
                f"Expected {len(PRICE_COLS)} close columns, found {len(close_idx)}"
            )
        name_to_idx = {feature_names[i]: i for i in close_idx}
        ordered_idx = [name_to_idx[name] for name in PRICE_COLS]
        prices = features[:, :, ordered_idx].astype(np.float32)

        keep_mask = np.ones(features.shape[2], dtype=bool)
        keep_mask[close_idx] = False
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
# Config
# ============================================================

@dataclass
class CoreConfig:
    """All env hyperparameters. Tweakable, but the defaults encode an
    institutional risk-management philosophy: rolling-window drawdowns,
    excess-relative-to-benchmark penalties, asymmetric upside/crisis-alpha
    rewards."""

    max_gross: float = 1.5
    initial_equity: float = 1.0
    episode_length: int = 384

    # Excess-drawdown windows and thresholds.
    # Penalty fires when (agent_dd - bench_dd) exceeds the threshold over the
    # window. Excess drawdown is the right metric: a 30% drawdown alongside a
    # 30% benchmark drawdown is acceptable; a 30% drawdown when benchmark is
    # flat is the manager's fault.
    dd_window_short: int = 21          # 1-month rolling
    dd_window_long: int = 63           # 1-quarter rolling
    dd_threshold_short: float = 0.05   # 5% excess dd over 1 month
    dd_threshold_long: float = 0.10    # 10% excess dd over 1 quarter
    lambda_dd_short: float = 10.0
    lambda_dd_long: float = 5.0

    # Trading cost
    lambda_turnover: float = 0.01

    # Asymmetric quarterly benchmark tracking.
    # No penalty/bonus until the rolling window has at least `benchmark_warmup`
    # days of data. Threshold is the symmetric tolerance band (±2% over the
    # quarter is the "no signal" zone).
    benchmark_window: int = 63
    benchmark_warmup: int = 21
    benchmark_threshold: float = 0.02

    # The four asymmetric lambdas.
    # Heavy weight (1.5) on the regime-correct extreme outcomes:
    #   - Failing to participate in a rally (career risk)
    #   - Generating crisis alpha (the prize)
    # Mild weight (0.3) on regime-incorrect outcomes:
    #   - Outperforming in a rally (alpha is nice but not required)
    #   - Underperforming in a selloff (forgivable, world is bad)
    lambda_upside_miss: float = 0.5
    lambda_upside_beat: float = 0.1
    lambda_downside_excess: float = 0.1
    lambda_crisis_alpha: float = 0.5


# ============================================================
# Portfolio core
# ============================================================

class PortfolioCore:
    """Stateful portfolio simulator with rolling-window risk tracking.

    Maintains parallel agent and benchmark equity curves so all risk metrics
    are computed *relative* to benchmark. Rolling windows mean intraday
    volatility doesn't trigger penalties — only sustained drawdowns do.
    """

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
        """Create empty rolling windows. Called from reset()."""
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

        # Agent and benchmark both start at the same equity, evolve in parallel
        self.equity = self.cfg.initial_equity
        self.bench_equity = self.cfg.initial_equity

        # Risk metrics, all initialized to no-stress
        self.excess_dd_short = 0.0
        self.excess_dd_long = 0.0
        self.quarterly_excess = 0.0
        self.quarterly_bench = 0.0

        self.weights = np.zeros(self.n_assets, dtype=np.float32)

        self._init_windows()
        # Seed the equity windows with starting equity so first-step drawdown is 0
        self.equity_window_short.append(self.equity)
        self.equity_window_long.append(self.equity)
        self.bench_equity_window_short.append(self.bench_equity)
        self.bench_equity_window_long.append(self.bench_equity)

    @property
    def steps_remaining(self) -> int:
        return self.t_end - self.t

    def portfolio_state(self) -> np.ndarray:
        """The portfolio state portion of the obs vector. 10 elements:
        equity, excess_dd_short, gross, net, w[4], quarterly_excess, quarterly_bench."""
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

    # ------------------------------------------------------------------
    # Action parsing — fixed projection (handles longs and shorts correctly)
    # ------------------------------------------------------------------

    def parse_hl_action(self, hl_action: np.ndarray) -> Tuple[float, float]:
        gross_raw, net_raw = float(hl_action[0]), float(hl_action[1])
        # Constrained gross: map [-1, +1] -> [0.8, 1.0]
        # Forces HL to always run a meaningfully-leveraged book; cannot
        # express defensiveness through size, only through direction (net).
        GROSS_MIN, GROSS_MAX = 0.8, 1.0
        target_gross = GROSS_MIN + (gross_raw + 1.0) / 2.0 * (GROSS_MAX - GROSS_MIN)
        target_net = net_raw * target_gross
        return target_gross, target_net

    def parse_ll_action(
        self,
        ll_action: np.ndarray,
        target_gross: float,
        target_net: float,
    ) -> np.ndarray:
        """Long/short book projection. Honors target_net exactly; gross is
        best-effort when LL signal lacks the necessary sign for shorts."""
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

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def apply_allocation(self, new_weights: np.ndarray) -> Tuple[float, bool, dict]:
        """Apply the new allocation, realizing tomorrow's return on it.

        Timing semantics (end-of-day rebalance):
          - Agent observes features[t] at end of day t (includes today's return)
          - Agent decides new_weights, the overnight position into day t+1
          - PL is realized when day t+1 closes: weights * returns[t+1]
        """
        old_weights = self.weights.copy()
        self.weights = new_weights.astype(np.float32)

        # CRITICAL: use NEXT day's return — the return realized on the position
        # the agent just took, not today's already-known return.
        next_t = self.t + 1
        if next_t >= len(self.returns):
            # End of data — terminate immediately, no return realized this step
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
            self.t = next_t  # advance for consistency
            return 0.0, True, info

        asset_returns = self.returns[next_t]
        portfolio_return = float(np.dot(self.weights, asset_returns))
        bench_return = float(self.benchmark_returns[next_t])

        self.equity *= 1.0 + portfolio_return
        self.bench_equity *= 1.0 + bench_return

        # Update rolling equity windows
        self.equity_window_short.append(self.equity)
        self.equity_window_long.append(self.equity)
        self.bench_equity_window_short.append(self.bench_equity)
        self.bench_equity_window_long.append(self.bench_equity)

        # Drawdowns from rolling-window peaks
        agent_peak_short = max(self.equity_window_short)
        agent_peak_long = max(self.equity_window_long)
        bench_peak_short = max(self.bench_equity_window_short)
        bench_peak_long = max(self.bench_equity_window_long)

        agent_dd_short = 1.0 - self.equity / agent_peak_short
        agent_dd_long = 1.0 - self.equity / agent_peak_long
        bench_dd_short = 1.0 - self.bench_equity / bench_peak_short
        bench_dd_long = 1.0 - self.bench_equity / bench_peak_long

        self.excess_dd_short = agent_dd_short - bench_dd_short
        self.excess_dd_long = agent_dd_long - bench_dd_long

        # Quarterly cumulative excess return + benchmark return
        excess = portfolio_return - bench_return
        self.excess_window.append(excess)
        self.bench_window.append(bench_return)
        self.quarterly_excess = sum(self.excess_window)
        self.quarterly_bench = sum(self.bench_window)

        # Turnover
        turnover = float(np.sum(np.abs(self.weights - old_weights)))

        reward = self._reward(portfolio_return, turnover)

        self.t += 1
        done = self.t >= self.t_end

        info = {
            "equity": self.equity,
            "bench_equity": self.bench_equity,
            "excess_dd_short": self.excess_dd_short,
            "excess_dd_long": self.excess_dd_long,
            "agent_dd_short": agent_dd_short,
            "bench_dd_short": bench_dd_short,
            "weights": self.weights.copy(),
            "turnover": turnover,
            "portfolio_return": portfolio_return,
            "benchmark_return": bench_return,
            "quarterly_excess": self.quarterly_excess,
            "quarterly_bench": self.quarterly_bench,
        }
        return reward, done, info
    def _reward(self, portfolio_return: float, turnover: float) -> float:
        # 1. Log growth — direct reward for absolute returns
        r = max(portfolio_return, -0.999)
        reward = float(np.log1p(r))

        # 2. Excess-drawdown penalties on rolling windows.
        #    Only fires when agent is in deeper drawdown than benchmark by more
        #    than the threshold. A 30% drawdown alongside a 30% benchmark
        #    drawdown produces zero penalty here.
        excess_dd_short_above = max(0.0, self.excess_dd_short - self.cfg.dd_threshold_short)
        reward -= self.cfg.lambda_dd_short * excess_dd_short_above ** 2

        excess_dd_long_above = max(0.0, self.excess_dd_long - self.cfg.dd_threshold_long)
        reward -= self.cfg.lambda_dd_long * excess_dd_long_above ** 2

        # 3. Turnover penalty
        reward -= self.cfg.lambda_turnover * turnover

        # 4. Asymmetric quarterly benchmark term.
        #    Doesn't fire until the rolling window has at least benchmark_warmup
        #    days of data. Two-sided asymmetry:
        #    - Bench up: failing to participate is heavily penalized, beating
        #      it gives a mild bonus.
        #    - Bench down: excess loss is mildly penalized, generating crisis
        #      alpha gives a heavy bonus.
        if len(self.excess_window) >= self.cfg.benchmark_warmup:
            excess_above_band = self.quarterly_excess - self.cfg.benchmark_threshold
            excess_below_band = -(self.quarterly_excess + self.cfg.benchmark_threshold)

            if self.quarterly_bench > 0:
                # Up market
                if excess_below_band > 0:
                    # Missed the rally — career risk
                    reward -= self.cfg.lambda_upside_miss * excess_below_band
                elif excess_above_band > 0:
                    # Outperformed in a rally — mild bonus
                    reward += self.cfg.lambda_upside_beat * excess_above_band
            else:
                # Down market
                if excess_below_band > 0:
                    # Underperformed in a selloff — mild penalty
                    reward -= self.cfg.lambda_downside_excess * excess_below_band
                elif excess_above_band > 0:
                    # Crisis alpha — the prize
                    reward += self.cfg.lambda_crisis_alpha * excess_above_band

        return float(reward)


# ============================================================
# Pool sampler
# ============================================================

class SyntheticPoolCoreSampler:
    """Acts like a PortfolioCore but each reset() picks a random path from
    the synthetic pool. Same interface, same risk tracking."""

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
# Environments — observation space accounts for the new state vector
# ============================================================

class LowLevelPortfolioEnv(gym.Env):
    """LL env. HL action is fixed (rule-based) during LL training.
    Obs = features (313) + portfolio_state (10) + hl_action (2) = 325 dim."""

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

        # 313 features + 10 portfolio state (equity, excess_dd, gross, net,
        # 4 weights, q_excess, q_bench) + 2 HL action = 325
        portfolio_state_dim = 4 + core.n_assets + 2  # equity, excess_dd, gross, net + n_assets weights + q_excess, q_bench
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


class HighLevelPortfolioEnv(gym.Env):
    """HL env. LL is frozen. Obs = features + portfolio_state = 323 dim."""

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
# Policy
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


# ============================================================
# Vec env helpers
# ============================================================

def make_vec_env(
    core_factory,
    env_class,
    n_envs: int = 8,
    use_subproc: bool = True,
    vecnormalize: bool = True,
    seed: int = 0,
    **env_kwargs,
):
    def _make(rank):
        def _init():
            core = core_factory(rank)
            return env_class(core, **env_kwargs)
        return _init

    fns = [_make(i) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if (use_subproc and n_envs > 1) else DummyVecEnv(fns)
    vec.seed(seed)

    if vecnormalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return vec
