from typing import Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from torch import nn

# ============================================================
# Shared portfolio simulator
# ============================================================

class PortfolioCore:
    def __init__(
        self,
        features,
        returns,
        benchmark_returns=None,
        n_sleeves=5,
        max_gross=1.5,
        initial_equity=1.0,
        dd_threshold=0.15,
        lambda_dd=10.0,
        lambda_turnover=0.01,
        lambda_down=0.5,
    ):
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.benchmark_returns = (
            benchmark_returns.astype(np.float32)
            if benchmark_returns is not None
            else np.zeros(len(features), dtype=np.float32)
        )

        self.n_steps = len(features)
        self.feature_dim = features.shape[1]
        self.n_assets = returns.shape[1]
        self.n_sleeves = n_sleeves
        self.max_gross = max_gross

        self.initial_equity = initial_equity
        self.dd_threshold = dd_threshold
        self.lambda_dd = lambda_dd
        self.lambda_turnover = lambda_turnover
        self.lambda_down = lambda_down

        self.reset()

    def reset(self):
        self.t = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.drawdown = 0.0
        self.benchmark_gap = 0.0
        self.sleeves = np.zeros((self.n_sleeves, self.n_assets), dtype=np.float32)
        self.weights = self.sleeves.mean(axis=0)

    def portfolio_state(self):
        gross = np.sum(np.abs(self.weights))
        net = np.sum(self.weights)

        return np.array(
            [
                self.equity,
                self.drawdown,
                gross,
                net,
                *self.weights,
                self.benchmark_gap,
            ],
            dtype=np.float32,
        )

    def obs(self):
        return np.concatenate(
            [self.features[self.t], self.portfolio_state()]
        ).astype(np.float32)

    def parse_hl_action(self, hl_action):
        gross_raw, net_raw = hl_action

        target_gross = (gross_raw + 1.0) / 2.0 * self.max_gross
        target_net = net_raw * target_gross

        return float(target_gross), float(target_net)

    def parse_ll_action(self, ll_action, target_gross, target_net):
        raw = np.asarray(ll_action, dtype=np.float32)

        abs_raw = np.abs(raw)
        if abs_raw.sum() < 1e-8:
            weights = np.zeros_like(raw)
        else:
            weights = abs_raw / abs_raw.sum() * target_gross
            weights *= np.sign(raw)

        current_net = weights.sum()
        weights += (target_net - current_net) / self.n_assets

        gross = np.sum(np.abs(weights))
        if gross > target_gross and gross > 1e-8:
            weights = weights / gross * target_gross

        return weights.astype(np.float32)

    def apply_allocation(self, sleeve_weights):
        old_weights = self.weights.copy()

        sleeve_idx = self.t % self.n_sleeves
        self.sleeves[sleeve_idx] = sleeve_weights
        self.weights = self.sleeves.mean(axis=0)

        asset_returns = self.returns[self.t]
        portfolio_return = float(np.dot(self.weights, asset_returns))

        cash_weight = 1.0 - np.sum(np.abs(self.weights))
        portfolio_return += cash_weight * 0.0

        self.equity *= 1.0 + portfolio_return
        self.peak_equity = max(self.peak_equity, self.equity)
        self.drawdown = 1.0 - self.equity / self.peak_equity

        turnover = float(np.sum(np.abs(self.weights - old_weights)))

        benchmark_return = float(self.benchmark_returns[self.t])
        relative_return = portfolio_return - benchmark_return
        self.benchmark_gap = 0.05 * relative_return + 0.95 * self.benchmark_gap

        reward = self.reward(portfolio_return, turnover)

        self.t += 1
        done = self.t >= self.n_steps - 1

        info = {
            "equity": self.equity,
            "drawdown": self.drawdown,
            "weights": self.weights.copy(),
            "turnover": turnover,
            "portfolio_return": portfolio_return,
        }

        return reward, done, info

    def reward(self, portfolio_return, turnover):
        reward = np.log1p(np.clip(portfolio_return, -0.999, None))

        dd_excess = max(0.0, self.drawdown - self.dd_threshold)
        reward -= self.lambda_dd * dd_excess**2
        reward -= self.lambda_turnover * turnover

        if self.benchmark_gap < 0:
            reward -= self.lambda_down * abs(self.benchmark_gap)

        return float(reward)


# ============================================================
# Low-level environment
# Train this first.
# HL is rule-based during LL training.
# ============================================================

class LowLevelPortfolioEnv(gym.Env):
    def __init__(self, core: PortfolioCore, fixed_hl_action=None):
        super().__init__()

        self.core = core
        self.fixed_hl_action = (
            np.array([0.33, 1.0], dtype=np.float32)
            if fixed_hl_action is None
            else fixed_hl_action
        )

        obs_dim = self.core.feature_dim + 1 + 1 + 1 + 1 + self.core.n_assets + 1 + 2

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.core.n_assets,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset()
        return self._get_obs(), {}

    def step(self, ll_action):
        target_gross, target_net = self.core.parse_hl_action(self.fixed_hl_action)

        sleeve_weights = self.core.parse_ll_action(
            ll_action,
            target_gross,
            target_net,
        )

        reward, done, info = self.core.apply_allocation(sleeve_weights)

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        return np.concatenate(
            [self.core.obs(), self.fixed_hl_action]
        ).astype(np.float32)


# ============================================================
# High-level environment
# Train this after LL is trained.
# LL model is frozen and used inside this environment.
# ============================================================

class HighLevelPortfolioEnv(gym.Env):
    def __init__(self, core: PortfolioCore, ll_model: PPO):
        super().__init__()

        self.core = core
        self.ll_model = ll_model

        obs_dim = self.core.feature_dim + 1 + 1 + 1 + 1 + self.core.n_assets + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset()
        return self.core.obs(), {}

    def step(self, hl_action):
        target_gross, target_net = self.core.parse_hl_action(hl_action)

        ll_obs = np.concatenate(
            [self.core.obs(), hl_action]
        ).astype(np.float32)

        ll_action, _ = self.ll_model.predict(ll_obs, deterministic=False)

        sleeve_weights = self.core.parse_ll_action(
            ll_action,
            target_gross,
            target_net,
        )

        reward, done, info = self.core.apply_allocation(sleeve_weights)
        info["hl_action"] = hl_action
        info["ll_action"] = ll_action

        return self.core.obs(), reward, done, False, info


class LayerNormMlpExtractor(nn.Module):
    def __init__(self, features_dim=299):
        super().__init__()
        self.policy_net = nn.Sequential(
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
        
        self.value_net = nn.Sequential(
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
        
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256
    
    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

class LayerNormActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = LayerNormMlpExtractor(
            self.features_dim
        )

def process_raw_df(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    price_cols = [
        "NVDA_close",
        "AMD_close",
        "SMH_close",
        "TLT_close"
    ]
    
    features_df = df.copy()
    features_df.drop(['date'] + price_cols, axis=1, inplace=True)
    features = features_df.to_numpy(dtype=np.float32)

    returns_df = df[price_cols].pct_change()
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    returns_df = returns_df.fillna(0.0)
    returns = returns_df.to_numpy(dtype=np.float32)
    
    return features, returns
    
    
# ============================================================
# Training example
# ============================================================

if __name__ == "__main__":
    n_days = 1_000
    feature_dim = 64
    n_assets = 4  # NVDA, AMD, SMH, TLT
    
    train_df = pd.read_csv('../data/proccessed/combined_w_cross_asset/train/RL_Final_Merged_train.csv')
    test_df = pd.read_csv('../data/proccessed/combined_w_cross_asset/test/RL_Final_Merged_test.csv')
    
    train_features, train_returns = process_raw_df(train_df)
    benchmark_returns = train_returns[:, :3].mean(axis=1)

    # ----------------------------
    # 1. Train low-level PPO first
    # ----------------------------

    ll_core = PortfolioCore(
        features=train_features,
        returns=train_returns,
        benchmark_returns=benchmark_returns,
    )

    ll_env = LowLevelPortfolioEnv(ll_core)

    check_env(ll_env, warn=True)

    ll_model = PPO(
        LayerNormActorCriticPolicy,
        ll_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    ll_model.learn(total_timesteps=100_000)
    ll_model.save("ppo_low_level_allocator")

    # ----------------------------
    # 2. Train high-level PPO second
    # ----------------------------

    hl_core = PortfolioCore(
        features=train_features,
        returns=train_returns,
        benchmark_returns=benchmark_returns,
    )

    hl_env = HighLevelPortfolioEnv(
        core=hl_core,
        ll_model=ll_model,
    )

    check_env(hl_env, warn=True)

    hl_model = PPO(
        LayerNormActorCriticPolicy,
        hl_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    hl_model.learn(total_timesteps=100_000)
    hl_model.save("ppo_high_level_controller")

    print("Training complete.")