"""
train/ll_finetune_real_no_regime.py

Fine-tunes the no-regime LL on real_train data. Mirrors the original
ll_random_hl_finetune.py protocol that produced the random-HL fine-tuned LL.

Run from project root:
    python train/ll_finetune_real_no_regime.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from project_config import PATHS
from env.portfolio_hrl_env_no_regime import (
    CoreConfig,
    LowLevelPortfolioEnvRandomHL,
    PortfolioCore,
    process_raw_df,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRETRAIN_DIR = PATHS.checkpoints / "ll_random_hl_synth_pretrain_no_regime"
PRETRAIN_MODEL = PRETRAIN_DIR / "ll_no_regime_pretrain_final.zip"
PRETRAIN_VECNORM = PRETRAIN_DIR / "ll_no_regime_pretrain_final_vecnorm.pkl"

REAL_EPISODE_LENGTH = 384
N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

NEW_LR = 3e-5
NEW_CLIP = 0.1
NEW_ENT_COEF = 0.03
NEW_N_EPOCHS = 4


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def build_train_vec_real(features, returns):
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _make(rank):
        def _init():
            core = PortfolioCore(
                features, returns, cfg=cfg,
                rng=np.random.default_rng(14000 + rank),
            )
            return LowLevelPortfolioEnvRandomHL(core, hl_seed=8000 + rank)
        return _init

    fns = [_make(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def build_eval_vec_real(features, returns):
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        return LowLevelPortfolioEnvRandomHL(core, hl_seed=42)

    vec = DummyVecEnv([_init])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
                       training=False)
    return vec


class SyncVecNormalizeCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            train_vec = self.model.get_vec_normalize_env()
            if train_vec is not None and hasattr(self.eval_env, "obs_rms"):
                self.eval_env.obs_rms = train_vec.obs_rms
                self.eval_env.ret_rms = train_vec.ret_rms
        return super()._on_step()


def main():
    print("=" * 80)
    print("LL fine-tune on real_train — NO regime probs")
    print("=" * 80)
    print(f"\nStarting from: {PRETRAIN_MODEL.name}")

    if not PRETRAIN_MODEL.exists():
        print(f"\nERROR: pretrain model not found: {PRETRAIN_MODEL}")
        sys.exit(1)
    if not PRETRAIN_VECNORM.exists():
        print(f"\nERROR: pretrain vecnorm not found: {PRETRAIN_VECNORM}")
        sys.exit(1)

    print("\nLoading real CSVs...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers on real_train...")
    train_vec = build_train_vec_real(feats_train, rets_train)
    eval_train_vec = build_eval_vec_real(feats_train, rets_train)
    eval_test_vec = build_eval_vec_real(feats_test, rets_test)

    print(f"\nLoading LL VecNormalize from pretrain: {PRETRAIN_VECNORM.name}")
    train_vec = VecNormalize.load(str(PRETRAIN_VECNORM), train_vec.venv)
    train_vec.training = True
    train_vec.norm_reward = False

    ckpt_dir = PATHS.checkpoints / "ll_finetune_real_no_regime"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "ll_finetune_real_no_regime"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    print("\nLoading LL model with hyperparameter overrides...")
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
        str(PRETRAIN_MODEL),
        env=train_vec,
        custom_objects=custom_objects,
        device="cpu",
        tensorboard_log=str(tb_dir),
    )
    model.ent_coef = NEW_ENT_COEF
    model.n_epochs = NEW_N_EPOCHS

    print(f"  LL params: {sum(p.numel() for p in model.policy.parameters()):,}")

    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_ll_ft_no_regime",
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

    callbacks = CallbackList([checkpoint_cb, eval_train_cb, eval_test_cb])

    print(f"\nStarting LL fine-tune: {TOTAL_TIMESTEPS:,} timesteps\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    finally:
        final_model_path = ckpt_dir / "ll_ft_no_regime_final.zip"
        final_vecnorm_path = ckpt_dir / "ll_ft_no_regime_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("LL fine-tune (no regime) complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()