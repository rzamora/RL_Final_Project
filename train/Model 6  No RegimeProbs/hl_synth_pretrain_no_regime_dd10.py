"""
train/hl_synth_pretrain_no_regime.py

HL synth pretrain on top of the no-regime fine-tuned LL.

Frozen LL: ll_ft_no_regime_final.zip (fine-tuned no-regime LL)
HL features: 309 (no regime probs) + 10 portfolio_state = 319-dim obs
HL action: 2 (gross_signal, net_signal)

Hyperparameters match the original hl_synth_pretrain_v2 (ent=0.03 variant)
to keep the comparison clean.

Run from project root:
    python train/hl_synth_pretrain_no_regime.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from project_config import PATHS
from env.portfolio_hrl_env_no_regime_dd10 import (
    CoreConfig,
    HighLevelPortfolioEnv,
    LayerNormActorCriticPolicy,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Frozen LL — use the fine-tuned no-regime LL
LL_MODEL_PATH = (PATHS.checkpoints / "ll_finetune_real_no_regime"
                 / "best_on_real_train" / "best_model.zip")
LL_VECNORM_PATH = (PATHS.checkpoints / "ll_finetune_real_no_regime"
                   / "ppo_ll_ft_no_regime_vecnormalize_100000_steps.pkl")

SYNTH_EPISODE_LENGTH = 383
REAL_EPISODE_LENGTH = 384

N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000
EVAL_FREQ = 50_000
POSTURE_EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

PPO_KWARGS = dict(
    learning_rate=3e-5,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.03,         # match HL v2 ent=0.03 variant
    n_epochs=4,
    device="cpu",
    seed=0,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Frozen LL helper
# ---------------------------------------------------------------------------

class _DummyObsEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, obs_dim):
        super().__init__()
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=-1e6, high=1e6,
                                              shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                         shape=(4,), dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, a):
        return (np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0, True, False, {})


def load_frozen_ll():
    print(f"  Loading frozen LL model: {LL_MODEL_PATH.name}")
    ll_model = PPO.load(str(LL_MODEL_PATH), device="cpu")

    print(f"  Loading LL VecNormalize: {LL_VECNORM_PATH.name}")
    dummy_env = DummyVecEnv([lambda: _DummyObsEnv(ll_model.observation_space.shape[0])])
    ll_vecnorm = VecNormalize.load(str(LL_VECNORM_PATH), dummy_env)
    ll_vecnorm.training = False

    expected_ll_obs_dim = 309 + 10 + 2  # features + portfolio_state + hl_action
    actual_ll_obs_dim = ll_model.observation_space.shape[0]
    assert actual_ll_obs_dim == expected_ll_obs_dim, (
        f"LL obs dim mismatch: model expects {actual_ll_obs_dim}, "
        f"no-regime LL should have {expected_ll_obs_dim}"
    )
    print(f"  LL obs dim verified: {actual_ll_obs_dim} (no regime probs)")

    def predict_fn(ll_obs_unnormalized: np.ndarray):
        obs_batch = ll_obs_unnormalized.reshape(1, -1).astype(np.float32)
        normalized = ll_vecnorm.normalize_obs(obs_batch)
        action, _ = ll_model.predict(normalized, deterministic=True)
        return action[0]

    return predict_fn


class FrozenLLAdapter:
    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def predict(self, obs, deterministic=True):
        action = self._predict_fn(obs)
        return action, None


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def build_train_vec(pool, ll_adapter):
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def _make(rank):
        def _init():
            sampler = SyntheticPoolCoreSampler(
                pool=pool, cfg=cfg,
                rng=np.random.default_rng(15000 + rank),
            )
            return HighLevelPortfolioEnv(sampler, ll_adapter)
        return _init

    fns = [_make(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def build_eval_vec_real(features, returns, ll_adapter):
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        return HighLevelPortfolioEnv(core, ll_adapter)

    vec = DummyVecEnv([_init])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
                       training=False)
    return vec


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class SyncVecNormalizeCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            train_vec = self.model.get_vec_normalize_env()
            if train_vec is not None and hasattr(self.eval_env, "obs_rms"):
                self.eval_env.obs_rms = train_vec.obs_rms
                self.eval_env.ret_rms = train_vec.ret_rms
        return super()._on_step()


class PerRegimePostureCallback(BaseCallback):
    """Posture diagnostic on real_test using true regime labels from the CSV."""

    def __init__(self, features, returns, regime_idx, ll_adapter,
                 eval_freq, train_vec_for_norm, verbose=0):
        super().__init__(verbose)
        self.features = features
        self.returns = returns
        self.regime_idx = regime_idx
        self.ll_adapter = ll_adapter
        self.eval_freq = eval_freq
        self.train_vec_for_norm = train_vec_for_norm
        self.regime_names = ["Bull", "Bear", "SevereBear", "Crisis"]

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)
        core = PortfolioCore(self.features, self.returns, cfg=cfg,
                              rng=np.random.default_rng(0))
        core.t_start = 0
        core.t_end = REAL_EPISODE_LENGTH
        core.t = 0
        core.equity = cfg.initial_equity
        core.bench_equity = cfg.initial_equity
        core._init_windows()
        core.equity_window_short.append(core.equity)
        core.equity_window_long.append(core.equity)
        core.bench_equity_window_short.append(core.bench_equity)
        core.bench_equity_window_long.append(core.bench_equity)

        env = HighLevelPortfolioEnv(core, self.ll_adapter)
        obs, _ = env.reset()

        gross_history = []
        net_history = []
        done = False
        while not done:
            obs_batch = obs.reshape(1, -1).astype(np.float32)
            obs_norm = self.train_vec_for_norm.normalize_obs(obs_batch)[0]
            action, _ = self.model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            gross, net = core.parse_hl_action(action)
            gross_history.append(gross)
            net_history.append(net)
            done = terminated or truncated

        gross_arr = np.array(gross_history)
        net_arr = np.array(net_history)
        T = len(gross_arr)
        regime_slice = self.regime_idx[:T]

        for i, name in enumerate(self.regime_names):
            mask = regime_slice == i
            if mask.sum() > 0:
                self.logger.record(f"posture/gross_{name}", float(gross_arr[mask].mean()))
                self.logger.record(f"posture/net_{name}",   float(net_arr[mask].mean()))
                self.logger.record(f"posture/n_{name}",     int(mask.sum()))

        self.logger.record("posture/gross_overall_mean", float(gross_arr.mean()))
        self.logger.record("posture/net_overall_mean",   float(net_arr.mean()))
        self.logger.record("posture/gross_std",          float(gross_arr.std()))
        self.logger.record("posture/net_std",            float(net_arr.std()))

        bull_mask = regime_slice == 0
        sb_mask = regime_slice == 2
        if bull_mask.sum() > 0 and sb_mask.sum() > 0:
            net_gap = float(net_arr[sb_mask].mean() - net_arr[bull_mask].mean())
            gross_gap = float(gross_arr[sb_mask].mean() - gross_arr[bull_mask].mean())
            self.logger.record("posture/net_gap_SB_minus_Bull", net_gap)
            self.logger.record("posture/gross_gap_SB_minus_Bull", gross_gap)

        if self.verbose > 0:
            print(f"\n[posture @ step {self.num_timesteps}] "
                  f"gross overall={gross_arr.mean():.3f} (std {gross_arr.std():.3f}), "
                  f"net overall={net_arr.mean():+.3f} (std {net_arr.std():.3f})")
            for i, name in enumerate(self.regime_names):
                mask = regime_slice == i
                if mask.sum() > 0:
                    print(f"  {name:<11s} n={mask.sum():>3d}  "
                          f"gross={gross_arr[mask].mean():.3f}  "
                          f"net={net_arr[mask].mean():+.3f}")

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("HL synth pretrain on no-regime LL")
    print("=" * 80)

    if not LL_MODEL_PATH.exists():
        print(f"\nERROR: LL model not found: {LL_MODEL_PATH}")
        sys.exit(1)
    if not LL_VECNORM_PATH.exists():
        print(f"\nERROR: LL vecnorm not found: {LL_VECNORM_PATH}")
        sys.exit(1)

    print("\nLoading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}  (expected (2000, 384, 309))")
    print(f"  returns std: {pool['returns'].std():.4f}")

    print("\nLoading real CSVs...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    # Use original CSV columns (still present even though stripped from features)
    # to compute regime labels for the posture diagnostic
    regime_cols = ["regime_prob_Bull", "regime_prob_Bear",
                   "regime_prob_SevereBear", "regime_prob_Crisis"]
    test_regime_idx = test_df[regime_cols].to_numpy().argmax(axis=1)

    print("\nLoading frozen LL (no-regime fine-tuned)...")
    ll_predict_fn = load_frozen_ll()
    ll_adapter = FrozenLLAdapter(ll_predict_fn)

    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers on synth pool...")
    train_vec = build_train_vec(pool, ll_adapter)

    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec_real(feats_train, rets_train, ll_adapter)
    eval_test_vec = build_eval_vec_real(feats_test, rets_test, ll_adapter)

    ckpt_dir = PATHS.checkpoints / "hl_synth_pretrain_no_regime_dd10"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "hl_synth_pretrain_no_regime_dd10"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    print("\nBuilding HL PPO with LayerNormActorCriticPolicy (fresh init)...")
    model = PPO(
        LayerNormActorCriticPolicy,
        train_vec,
        tensorboard_log=str(tb_dir),
        **PPO_KWARGS,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  HL policy params: {n_params:,}")
    print(f"  HL obs space:    {model.observation_space.shape}  (expected (319,))")
    print(f"  HL action space: {model.action_space.shape}  (expected (2,))")

    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)
    posture_freq_pw = max(POSTURE_EVAL_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_hl_no_regime",
        save_vecnormalize=True,
    )
    eval_train_cb = SyncVecNormalizeCallback(
        eval_env=eval_train_vec, n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq_pw,
        log_path=str(tb_dir / "eval_real_train"),
        best_model_save_path=str(ckpt_dir / "best_on_real_train"),
        deterministic=True, render=False, verbose=1,
    )
    eval_test_cb = SyncVecNormalizeCallback(
        eval_env=eval_test_vec, n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq_pw,
        log_path=str(tb_dir / "eval_real_test"),
        best_model_save_path=str(ckpt_dir / "best_on_real_test"),
        deterministic=True, render=False, verbose=1,
    )
    posture_cb = PerRegimePostureCallback(
        features=feats_test, returns=rets_test,
        regime_idx=test_regime_idx,
        ll_adapter=ll_adapter,
        eval_freq=posture_freq_pw,
        train_vec_for_norm=train_vec,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb, eval_train_cb, eval_test_cb, posture_cb])

    print(f"\nStarting HL training: {TOTAL_TIMESTEPS:,} timesteps\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    finally:
        final_model_path = ckpt_dir / "hl_no_regime_final.zip"
        final_vecnorm_path = ckpt_dir / "hl_no_regime_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("HL synth pretrain (no regime) complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()