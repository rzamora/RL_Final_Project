"""
train/ll_random_hl_synth_pretrain_no_regime.py

LL synth pretrain with random HL actions, on the no-regime feature set.

Mirrors the original ll_random_hl_synth_pretrain.py but uses the
portfolio_hrl_env_no_regime env classes (309 features, no regime probs).

Run from project root:
    python train/ll_random_hl_synth_pretrain_no_regime.py
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
    LayerNormActorCriticPolicy,
    LowLevelPortfolioEnv,
    LowLevelPortfolioEnvRandomHL,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYNTH_EPISODE_LENGTH = 383
REAL_EPISODE_LENGTH = 384

N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000
EVAL_FREQ = 200_000
N_EVAL_EPISODES = 10

PPO_KWARGS = dict(
    learning_rate=1e-4,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    n_epochs=4,
    device="cpu",
    seed=0,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def build_train_vec(pool):
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def _make(rank):
        def _init():
            sampler = SyntheticPoolCoreSampler(
                pool=pool, cfg=cfg,
                rng=np.random.default_rng(13000 + rank),
            )
            return LowLevelPortfolioEnvRandomHL(sampler, hl_seed=7919 + rank)
        return _init

    fns = [_make(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def build_eval_vec_real(features, returns):
    """Eval env uses random HL actions per episode, same as training."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        return LowLevelPortfolioEnvRandomHL(core, hl_seed=42)

    vec = DummyVecEnv([_init])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
                       training=False)
    return vec


# ---------------------------------------------------------------------------
# Sync vecnormalize during eval
# ---------------------------------------------------------------------------

class SyncVecNormalizeCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            train_vec = self.model.get_vec_normalize_env()
            if train_vec is not None and hasattr(self.eval_env, "obs_rms"):
                self.eval_env.obs_rms = train_vec.obs_rms
                self.eval_env.ret_rms = train_vec.ret_rms
        return super()._on_step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("LL synth pretrain — random HL, NO regime probs")
    print("=" * 80)

    print("\nLoading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}  (expected (2000, 384, 309))")
    print(f"  returns std: {pool['returns'].std():.4f}")
    if pool["returns"].std() > 0.5:
        raise RuntimeError("Synth returns not in fractional units")

    print("\nLoading real CSVs...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers (random HL, no regime)...")
    train_vec = build_train_vec(pool)

    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec_real(feats_train, rets_train)
    eval_test_vec = build_eval_vec_real(feats_test, rets_test)

    ckpt_dir = PATHS.checkpoints / "ll_random_hl_synth_pretrain_no_regime"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "ll_random_hl_synth_pretrain_no_regime"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    print("\nBuilding LL PPO with LayerNormActorCriticPolicy (fresh init)...")
    model = PPO(
        LayerNormActorCriticPolicy,
        train_vec,
        tensorboard_log=str(tb_dir),
        **PPO_KWARGS,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  LL policy params: {n_params:,}")
    print(f"  LL obs space:    {model.observation_space.shape}  (expected (321,))")
    print(f"  LL action space: {model.action_space.shape}  (expected (4,))")

    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_ll_no_regime",
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

    print(f"\nStarting LL training: {TOTAL_TIMESTEPS:,} timesteps")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    finally:
        final_model_path = ckpt_dir / "ll_no_regime_pretrain_final.zip"
        final_vecnorm_path = ckpt_dir / "ll_no_regime_pretrain_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("LL synth pretrain (no regime) complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()