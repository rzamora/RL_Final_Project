"""
Fixed and improved version of portfolio_hrl_env.py.

Changes vs original (see 01_code_review.md for details):

  Bug 1 — process_raw_df schema now matches synthetic pool (317 features, no
          *_close drop). Returns are computed from the price columns separately.
  Bug 2 — parse_ll_action uses long/short-book projection. Gross and net
          targets are exactly satisfied (up to float precision).
  Bug 3 — Sleeve mechanism removed for v1. The README's sleeve concept can be
          re-added later; for now we have an honest "full portfolio output"
          policy with turnover penalty doing the smoothing job.
  Issue 4 — reset() supports random episode starts with configurable length.
  Issue 11 — done condition uses all rows.
  Issue 14 — RNG plumbed through.

The training script at the bottom shows the recommended pipeline:
   synthetic pre-train → real fine-tune.

Drop-in compatible with stable-baselines3 PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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

    Why we drop the *_close columns from features:
        Absolute price levels are a temporal fingerprint. NVDA at $0.11 in 2004
        and at $140 in 2024 lets the policy network learn "where in time am I"
        directly from the price magnitude, which leaks the calendar position
        into the policy. A regime-aware policy should be invariant to absolute
        price level — only relative changes (returns, ratios) carry signal.

    The four *_close columns are kept on the side (returned as `prices`) for
    use by anything downstream that needs them (return computation already
    does pct_change separately, but other consumers like benchmark code or
    diagnostic plots may want raw prices).

    Note: *_kronos_close_d5 is NOT dropped. Despite the name, it stores the
    Kronos-predicted return as a fraction (e.g. -0.0068, +0.012), not a
    dollar price. So it doesn't leak calendar position.

    features: (T, 313) — all CSV columns except `date` and the 4 *_close cols.
    returns:  (T, 4)   — pct_change of the four close columns, NaN→0.
    prices:   (T, 4)   — raw close prices, kept for downstream use.
    """
    price_cols = list(PRICE_COLS)

    feats_df = df.drop(columns=["date"] + price_cols)
    features = feats_df.to_numpy(dtype=np.float32)

    prices = df[price_cols].to_numpy(dtype=np.float32)

    rets_df = df[price_cols].pct_change()
    rets_df = rets_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns = rets_df.to_numpy(dtype=np.float32)

    # Defensive NaN scrub on features. Some technical indicators have small
    # warmup windows that may not have been trimmed.
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, returns, prices


def load_synthetic_pool(
    npz_path: str,
    drop_close_features: bool = True,
) -> dict:
    """
    Load the synthetic pool and (by default) strip the *_close columns from
    the feature tensor — same reason as in process_raw_df: absolute price
    levels would leak calendar position into the policy. The close prices
    remain available via pool['prices'] for downstream use.

    The pool's `feature_names` array tells us which columns to drop.

    Returns a dict with:
        features: (N, T, 313) if drop_close_features else (N, T, 317)
        returns:  (N, T, 4)
        prices:   (N, T, 4) — extracted from the *_close feature columns
                              before they were dropped (so the agent's env
                              has a price series even on synthetic paths)
        regimes:  (N, T) or None
        feature_names: list of feature names AFTER dropping (if any)
    """
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]              # (N, T, 317)
    returns = data["returns"]                # (N, T, 4)
    feature_names = list(data["feature_names"]) if "feature_names" in data.files else None

    prices = None

    if drop_close_features and feature_names is not None:
        close_idx = [i for i, n in enumerate(feature_names) if n in PRICE_COLS]
        if len(close_idx) != len(PRICE_COLS):
            raise ValueError(
                f"Expected {len(PRICE_COLS)} close columns in pool features, "
                f"found {len(close_idx)}. Pool schema may not match CSV."
            )
        # Reorder close_idx to match PRICE_COLS order so prices columns line up
        # with returns columns.
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
# Portfolio core
# ============================================================

@dataclass
class CoreConfig:
    max_gross: float = 1.5
    initial_equity: float = 1.0
    dd_threshold: float = 0.15
    lambda_dd: float = 10.0
    lambda_turnover: float = 0.01
    lambda_down: float = 0.5
    episode_length: int = 384


class PortfolioCore:
    """
    Stateful portfolio simulator. No sleeves in this version (see Bug 3 in
    01_code_review.md). The LL action is the full target portfolio.

    Random episode starts: when reset() is called, picks a random window of
    length cfg.episode_length within the available data.
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
            # Default: equal-weight on first 3 assets (NVDA, AMD, SMH).
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

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_start = self.full_n_steps - self.cfg.episode_length - 1
        self.t_start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self.t_end = self.t_start + self.cfg.episode_length
        self.t = self.t_start

        self.equity = self.cfg.initial_equity
        self.peak_equity = self.cfg.initial_equity
        self.drawdown = 0.0
        self.benchmark_gap = 0.0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)

    @property
    def steps_remaining(self) -> int:
        return self.t_end - self.t

    def portfolio_state(self) -> np.ndarray:
        gross = float(np.sum(np.abs(self.weights)))
        net = float(np.sum(self.weights))
        return np.array(
            [self.equity, self.drawdown, gross, net, *self.weights, self.benchmark_gap],
            dtype=np.float32,
        )

    def obs(self) -> np.ndarray:
        return np.concatenate([self.features[self.t], self.portfolio_state()]).astype(np.float32)

    # ------------------------------------------------------------------
    # Action parsing — fixed projection
    # ------------------------------------------------------------------

    def parse_hl_action(self, hl_action: np.ndarray) -> Tuple[float, float]:
        """HL action ∈ [-1, 1]^2 → (target_gross, target_net)."""
        gross_raw, net_raw = float(hl_action[0]), float(hl_action[1])
        target_gross = (gross_raw + 1.0) / 2.0 * self.cfg.max_gross  # ∈ [0, max_gross]
        target_net = net_raw * target_gross                          # ∈ [-target_gross, target_gross]
        return target_gross, target_net

    def parse_ll_action(
        self,
        ll_action: np.ndarray,
        target_gross: float,
        target_net: float,
    ) -> np.ndarray:
        """
        Long/short book projection. After this:
          |w|.sum() == target_gross   (exact, up to float)
          w.sum()   == target_net     (exact, up to float)
        unless the agent's chosen direction can't satisfy both — in which case
        target_net is best-effort.
        """
        raw = np.asarray(ll_action, dtype=np.float64)
        target_gross = float(np.clip(target_gross, 0.0, self.cfg.max_gross))
        target_net = float(np.clip(target_net, -target_gross, target_gross))

        long_gross = 0.5 * (target_gross + target_net)   # ≥ 0
        short_gross = 0.5 * (target_gross - target_net)  # ≥ 0

        pos = np.maximum(raw, 0.0)
        neg = np.maximum(-raw, 0.0)
        pos_sum = pos.sum()
        neg_sum = neg.sum()

        # Build long and short books separately, fall back to uniform within
        # each side when the agent provides no signal on that side.
        if pos_sum > 1e-8:
            long_book = (pos / pos_sum) * long_gross
        elif long_gross > 0:
            long_book = np.full(self.n_assets, long_gross / self.n_assets)
        else:
            long_book = np.zeros(self.n_assets)

        if neg_sum > 1e-8:
            short_book = (neg / neg_sum) * short_gross
        elif short_gross > 0:
            short_book = np.full(self.n_assets, short_gross / self.n_assets)
        else:
            short_book = np.zeros(self.n_assets)

        return (long_book - short_book).astype(np.float32)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def apply_allocation(self, new_weights: np.ndarray) -> Tuple[float, bool, dict]:
        old_weights = self.weights.copy()
        self.weights = new_weights.astype(np.float32)

        asset_returns = self.returns[self.t]
        portfolio_return = float(np.dot(self.weights, asset_returns))

        # Cash earns 0 here; can be replaced with rf_daily if desired.
        # cash_weight = 1.0 - np.sum(np.abs(self.weights))

        self.equity *= 1.0 + portfolio_return
        self.peak_equity = max(self.peak_equity, self.equity)
        self.drawdown = 1.0 - self.equity / self.peak_equity

        turnover = float(np.sum(np.abs(self.weights - old_weights)))

        bench_return = float(self.benchmark_returns[self.t])
        relative = portfolio_return - bench_return
        self.benchmark_gap = 0.05 * relative + 0.95 * self.benchmark_gap

        reward = self._reward(portfolio_return, turnover)

        self.t += 1
        done = self.t >= self.t_end

        info = {
            "equity": self.equity,
            "drawdown": self.drawdown,
            "weights": self.weights.copy(),
            "turnover": turnover,
            "portfolio_return": portfolio_return,
            "benchmark_return": bench_return,
        }
        return reward, done, info

    def _reward(self, portfolio_return: float, turnover: float) -> float:
        # log growth, capped to avoid -inf at full loss
        r = max(portfolio_return, -0.999)
        reward = float(np.log1p(r))

        # convex drawdown penalty above threshold
        dd_excess = max(0.0, self.drawdown - self.cfg.dd_threshold)
        reward -= self.cfg.lambda_dd * dd_excess ** 2

        # turnover penalty
        reward -= self.cfg.lambda_turnover * turnover

        # asymmetric benchmark penalty (downside only — README's λ_up is omitted
        # by design; see Issue 7 in the code review)
        if self.benchmark_gap < 0:
            reward -= self.cfg.lambda_down * abs(self.benchmark_gap)

        return float(reward)


# ============================================================
# Pool sampler — wraps a synthetic pool as a sequence of cores
# ============================================================

class SyntheticPoolCoreSampler:
    """
    Behaves like a PortfolioCore but on each reset, picks a random path from
    the synthetic pool to use as the underlying data. Implements the same
    interface so the env code doesn't change.
    """

    def __init__(
        self,
        pool: dict,
        cfg: Optional[CoreConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.pool_features = pool["features"]   # (N, T, 317)
        self.pool_returns = pool["returns"]     # (N, T, 4)
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

        # NaN scrub (defensive — pool is reportedly NaN-free, but cheap insurance)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        self._core = PortfolioCore(feats, rets, cfg=self.cfg, rng=self.rng)
        # PortfolioCore.__init__ already calls reset() which picks a random
        # window within the path. For pool paths of length 384 with episode
        # length 384, that window is the whole path.

    def __getattr__(self, name):
        # Delegate everything to the inner core
        return getattr(self._core, name)


# ============================================================
# Environments
# ============================================================

class LowLevelPortfolioEnv(gym.Env):
    """
    LL env. HL action is fixed (rule-based) during LL training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        core,                              # PortfolioCore or SyntheticPoolCoreSampler
        fixed_hl_action: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.core = core
        self.fixed_hl_action = (
            np.array([0.33, 0.5], dtype=np.float32)
            if fixed_hl_action is None
            else np.asarray(fixed_hl_action, dtype=np.float32)
        )

        # core.feature_dim + portfolio_state(4 + n_assets + 1) + hl_action(2)
        obs_dim = core.feature_dim + 4 + core.n_assets + 1 + 2
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
    """
    HL env. LL is frozen and used inside the env.
    """

    metadata = {"render_modes": []}

    def __init__(self, core, ll_model: PPO):
        super().__init__()
        self.core = core
        self.ll_model = ll_model

        obs_dim = core.feature_dim + 4 + core.n_assets + 1
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
# Policy with LayerNorm
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
# Vectorized env builders
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
    """
    core_factory: callable(rank) -> PortfolioCore | SyntheticPoolCoreSampler
    env_class:    LowLevelPortfolioEnv or HighLevelPortfolioEnv
    """

    def _make(rank):
        def _init():
            core = core_factory(rank)
            env = env_class(core, **env_kwargs)
            return env
        return _init

    fns = [_make(i) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if (use_subproc and n_envs > 1) else DummyVecEnv(fns)
    vec.seed(seed)

    if vecnormalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return vec


# ============================================================
# Training pipeline (illustrative)
# ============================================================

def train_pipeline(
    train_csv_path: str,
    synthetic_pool_path: str,
    out_dir: str = "./checkpoints",
    seed: int = 0,
    pretrain_steps: int = 2_000_000,
    finetune_steps: int = 500_000,
    n_envs_pretrain: int = 8,
    n_envs_finetune: int = 4,
):
    """
    Phase 1: synthetic pre-train  (LL then HL)
    Phase 2: real fine-tune       (LL then HL)

    Run 5 seeds for proper variance estimation.
    """
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    pool = load_synthetic_pool(synthetic_pool_path, drop_close_features=True)
    train_df = pd.read_csv(train_csv_path)
    train_features, train_returns, train_prices = process_raw_df(train_df)

    # Sanity check: synthetic features and real features must have the same
    # second dimension after dropping closes. If this fails, the synthetic
    # pool was built with a different feature set than the CSV.
    assert pool["features"].shape[2] == train_features.shape[1], (
        f"Feature-dim mismatch: pool has {pool['features'].shape[2]}, "
        f"real has {train_features.shape[1]}. Both should be 313."
    )

    # Train/val split: hold out last 252 days of train as validation
    val_split = len(train_features) - 252
    train_feat_split = train_features[:val_split]
    train_ret_split = train_returns[:val_split]

    cfg = CoreConfig(episode_length=384)

    # ============================================================
    # PHASE 1A: LL on synthetic
    # ============================================================
    def synth_core(rank):
        return SyntheticPoolCoreSampler(
            pool, cfg=cfg, rng=np.random.default_rng(seed * 1000 + rank)
        )

    ll_env = make_vec_env(
        synth_core, LowLevelPortfolioEnv,
        n_envs=n_envs_pretrain, vecnormalize=True, seed=seed,
    )

    ll_model = PPO(
        LayerNormActorCriticPolicy,
        ll_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
    )
    ll_model.learn(total_timesteps=pretrain_steps)
    ll_model.save(f"{out_dir}/ll_synth_pretrain_seed{seed}")
    ll_env.save(f"{out_dir}/ll_synth_vecnorm_seed{seed}.pkl")

    # ============================================================
    # PHASE 1B: HL on synthetic, frozen LL
    # ============================================================
    hl_env_synth = make_vec_env(
        synth_core, HighLevelPortfolioEnv,
        n_envs=n_envs_pretrain, vecnormalize=True, seed=seed + 100,
        ll_model=ll_model,
    )

    hl_model = PPO(
        LayerNormActorCriticPolicy,
        hl_env_synth,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed + 100,
    )
    hl_model.learn(total_timesteps=pretrain_steps // 2)
    hl_model.save(f"{out_dir}/hl_synth_pretrain_seed{seed}")
    hl_env_synth.save(f"{out_dir}/hl_synth_vecnorm_seed{seed}.pkl")

    # ============================================================
    # PHASE 2A: LL fine-tune on real
    # ============================================================
    def real_core(rank):
        return PortfolioCore(
            train_feat_split, train_ret_split,
            cfg=cfg, rng=np.random.default_rng(seed * 2000 + rank),
        )

    ll_env_real = make_vec_env(
        real_core, LowLevelPortfolioEnv,
        n_envs=n_envs_finetune, vecnormalize=True, seed=seed + 200,
    )
    # Lower LR + clip range for fine-tuning
    ll_model.set_env(ll_env_real)
    ll_model.learning_rate = 1e-4
    ll_model.clip_range = lambda _: 0.1
    ll_model.learn(total_timesteps=finetune_steps, reset_num_timesteps=False)
    ll_model.save(f"{out_dir}/ll_real_finetune_seed{seed}")
    ll_env_real.save(f"{out_dir}/ll_real_vecnorm_seed{seed}.pkl")

    # ============================================================
    # PHASE 2B: HL fine-tune on real
    # ============================================================
    hl_env_real = make_vec_env(
        real_core, HighLevelPortfolioEnv,
        n_envs=n_envs_finetune, vecnormalize=True, seed=seed + 300,
        ll_model=ll_model,
    )
    hl_model.set_env(hl_env_real)
    hl_model.learning_rate = 1e-4
    hl_model.clip_range = lambda _: 0.1
    hl_model.learn(total_timesteps=finetune_steps // 2, reset_num_timesteps=False)
    hl_model.save(f"{out_dir}/hl_real_finetune_seed{seed}")
    hl_env_real.save(f"{out_dir}/hl_real_vecnorm_seed{seed}.pkl")

    print(f"[seed {seed}] training pipeline complete.")
    return ll_model, hl_model


if __name__ == "__main__":
    # Single-seed example. For real results, run 5 seeds.
    train_pipeline(
        train_csv_path="data/proccessed/combined_w_cross_asset/train/RL_Final_Merged_train.csv",
        synthetic_pool_path="data/synthetic/pools/synthetic_pool_production_n2000_seed43.npz",
        out_dir="./checkpoints/seed0",
        seed=0,
        pretrain_steps=2_000_000,
        finetune_steps=500_000,
    )
