"""
train/hl_finetune_real_unconstrained.py

HL fine-tune on real_train data, starting from the HL ent=0.03 unconstrained
1M synth pretrain endpoint.

Mirrors the LL fine-tune approach (train/finetune_real.py):
  - Lower lr (3e-5, already conservative)
  - Tighter clip_range (0.05, was 0.1)
  - Lower ent_coef (0.01, was 0.03)
  - 200k steps (vs 1M synth pretrain)
  - Train on real CSV, eval on both real_train and real_test

Frozen LL: ll_random_hl_finetune (the random-HL fine-tuned LL, same as
during synth pretrain). LL is unchanged during HL fine-tune.

Run from project root:
    python train/hl_finetune_real_unconstrained.py
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
from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    HighLevelPortfolioEnv,
    PortfolioCore,
    process_raw_df,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Starting checkpoint: HL ent=0.03 unconstrained 1M final
HL_PRETRAIN_DIR = PATHS.checkpoints / "hl_synth_pretrain_v2"
HL_MODEL_PATH = HL_PRETRAIN_DIR / "hl_v2_final.zip"
HL_VECNORM_PATH = HL_PRETRAIN_DIR / "hl_v2_final_vecnorm.pkl"

# Frozen LL — same as used during HL synth pretrain
LL_MODEL_PATH = (PATHS.checkpoints / "ll_random_hl_finetune"
                 / "ll_random_hl_ft_final.zip")
LL_VECNORM_PATH = (PATHS.checkpoints / "ll_random_hl_finetune"
                   / "ll_random_hl_ft_final_vecnorm.pkl")

REAL_EPISODE_LENGTH = 384

N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 50_000
POSTURE_EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

# Fine-tune hyperparameter overrides
NEW_LR = 3e-5            # same as synth pretrain (already conservative)
NEW_CLIP = 0.05          # ↓ from 0.1 — refine, don't transform
NEW_ENT_COEF = 0.01      # ↓ from 0.03 — less exploration during fine-tune
NEW_N_EPOCHS = 4         # unchanged


# ---------------------------------------------------------------------------
# Frozen LL helper (same as v2 synth pretrain)
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
# Env factories — REAL TRAIN data, not synth pool
# ---------------------------------------------------------------------------

def build_train_vec_real(features, returns, ll_adapter):
    """Training envs use real_train CSV (not synth pool)."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _make(rank):
        def _init():
            core = PortfolioCore(
                features, returns, cfg=cfg,
                rng=np.random.default_rng(8000 + rank),
            )
            return HighLevelPortfolioEnv(core, ll_adapter)
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
    """Same posture diagnostic as synth pretrain."""

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
    print("HL fine-tune on real_train — UNCONSTRAINED gross")
    print("=" * 80)
    print(f"\nStarting from: {HL_MODEL_PATH.name}")
    print(f"Frozen LL:     {LL_MODEL_PATH.name}")

    if not HL_MODEL_PATH.exists():
        print(f"\nERROR: HL model not found: {HL_MODEL_PATH}")
        sys.exit(1)
    if not HL_VECNORM_PATH.exists():
        print(f"\nERROR: HL vecnorm not found: {HL_VECNORM_PATH}")
        sys.exit(1)
    if not LL_MODEL_PATH.exists():
        print(f"\nERROR: LL model not found: {LL_MODEL_PATH}")
        sys.exit(1)

    print("\nLoading real CSVs...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    regime_cols = ["regime_prob_Bull", "regime_prob_Bear",
                   "regime_prob_SevereBear", "regime_prob_Crisis"]
    test_regime_idx = test_df[regime_cols].to_numpy().argmax(axis=1)

    print("\nLoading frozen LL...")
    ll_predict_fn = load_frozen_ll()
    ll_adapter = FrozenLLAdapter(ll_predict_fn)

    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers on REAL TRAIN data...")
    train_vec = build_train_vec_real(feats_train, rets_train, ll_adapter)

    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec_real(feats_train, rets_train, ll_adapter)
    eval_test_vec = build_eval_vec_real(feats_test, rets_test, ll_adapter)

    # Load HL training VecNormalize stats into the new train_vec
    print(f"\nLoading HL VecNormalize from pretrain: {HL_VECNORM_PATH.name}")
    train_vec = VecNormalize.load(str(HL_VECNORM_PATH), train_vec.venv)
    train_vec.training = True  # let it adapt to real_train distributions
    train_vec.norm_reward = False

    ckpt_dir = PATHS.checkpoints / "hl_finetune_real_unconstrained"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "hl_finetune_real_unconstrained"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    print("\nLoading HL model with hyperparameter overrides...")
    print(f"  learning_rate: {NEW_LR}")
    print(f"  clip_range:    {NEW_CLIP}")
    print(f"  ent_coef:      {NEW_ENT_COEF}")
    print(f"  n_epochs:      {NEW_N_EPOCHS}")

    custom_objects = {
        "learning_rate": NEW_LR,
        "lr_schedule": lambda _: NEW_LR,
        "clip_range": lambda _: NEW_CLIP,
    }
    model = PPO.load(
        str(HL_MODEL_PATH),
        env=train_vec,
        custom_objects=custom_objects,
        device="cpu",
        tensorboard_log=str(tb_dir),
    )
    model.ent_coef = NEW_ENT_COEF
    model.n_epochs = NEW_N_EPOCHS

    print(f"  HL params: {sum(p.numel() for p in model.policy.parameters()):,}")

    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)
    posture_freq_pw = max(POSTURE_EVAL_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_hl_ft_uc",
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

    print(f"\nStarting HL fine-tune: {TOTAL_TIMESTEPS:,} timesteps")
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} ({ckpt_freq_pw}/worker)")
    print(f"  Eval every {EVAL_FREQ:,} ({eval_freq_pw}/worker), "
          f"{N_EVAL_EPISODES} episodes")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True,  # restart counter for clean fine-tune logs
        )
    finally:
        final_model_path = ckpt_dir / "hl_ft_uc_final.zip"
        final_vecnorm_path = ckpt_dir / "hl_ft_uc_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("HL fine-tune (unconstrained) complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()