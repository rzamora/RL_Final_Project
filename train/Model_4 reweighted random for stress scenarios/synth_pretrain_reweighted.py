"""
train/synth_pretrain_reweighted.py

LL-only synth pretrain (1M steps) with regime-reweighted path sampling.

Same hyperparameters and structure as the original train/synth_pretrain.py
(fixed HL action [0.33, 0.5], 8 SubprocVecEnv workers, lr=1e-4, n_epochs=4
which is the v3 lesson from the original n_epochs=10 instability) but uses
ReweightedSyntheticPoolCoreSampler so the agent sees:
    Bull-modal    25%
    Bear-modal    25%
    SevereBear    35%
    Crisis        15%
of training paths instead of the natural pool distribution
    Bull 48%, Bear 36%, SB 11%, Crisis 5%.

Run from project root:
    python train/synth_pretrain_reweighted.py
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
from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    LayerNormActorCriticPolicy,
    LowLevelPortfolioEnv,
    PortfolioCore,
    load_synthetic_pool,
    process_raw_df,
)
from env.portfolio_hrl_env_reweighted import (
    ReweightedSyntheticPoolCoreSampler,
    DEFAULT_TARGET_DISTRIBUTION,
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
    ent_coef=0.02,
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

    def core_factory(rank: int):
        sampler = ReweightedSyntheticPoolCoreSampler(
            pool=pool, cfg=cfg,
            rng=np.random.default_rng(13000 + rank),
            target_distribution=DEFAULT_TARGET_DISTRIBUTION,
        )
        return LowLevelPortfolioEnv(sampler)  # uses default fixed_hl_action [0.33, 0.5]

    fns = [lambda i=i: core_factory(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def build_eval_vec(features, returns):
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        return LowLevelPortfolioEnv(core)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("LL-only synth pretrain with REWEIGHTED path sampling — 1M steps")
    print("=" * 80)
    print()
    print("Target modal-regime distribution:")
    for r, name in enumerate(["Bull", "Bear", "SevereBear", "Crisis"]):
        print(f"  {name:<11s} {DEFAULT_TARGET_DISTRIBUTION[r]:.0%}")
    print()

    # ---------- Load synth pool ----------
    print("Loading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}")
    print(f"  returns:  {pool['returns'].shape}  std={pool['returns'].std():.4f}")
    if pool.get("regimes") is None:
        raise RuntimeError("Pool dict missing 'regimes' — cannot reweight.")

    # ---------- Sanity: instantiate one sampler to log effective distribution ----------
    print()
    cfg_check = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)
    test_sampler = ReweightedSyntheticPoolCoreSampler(
        pool=pool, cfg=cfg_check,
        rng=np.random.default_rng(0),
        target_distribution=DEFAULT_TARGET_DISTRIBUTION,
    )
    print("Pool natural distribution (modal-regime per path):")
    for r, name in enumerate(["Bull", "Bear", "SevereBear", "Crisis"]):
        nat = test_sampler.pool_modal_distribution.get(r, 0.0)
        n_paths_in_group = len(test_sampler._path_indices_by_regime[r])
        print(f"  {name:<11s} {nat:.1%}  ({n_paths_in_group} paths)")
    print("Effective sampling weights (after handling empty groups):")
    for r in test_sampler.effective_distribution:
        name = ["Bull", "Bear", "SevereBear", "Crisis"][r]
        print(f"  {name:<11s} {test_sampler.effective_distribution[r]:.0%}")
    print()

    # ---------- Real CSVs for eval ----------
    print("Loading real CSVs for periodic eval...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    # ---------- Build envs ----------
    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers (synth, reweighted)...")
    train_vec = build_train_vec(pool)
    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec(feats_train, rets_train)
    eval_test_vec = build_eval_vec(feats_test, rets_test)

    # ---------- Output paths ----------
    ckpt_dir = PATHS.checkpoints / "synth_pretrain_reweighted"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "synth_pretrain_reweighted"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    # ---------- PPO ----------
    print("\nBuilding PPO with LayerNormActorCriticPolicy...")
    model = PPO(
        LayerNormActorCriticPolicy,
        train_vec,
        tensorboard_log=str(tb_dir),
        **PPO_KWARGS,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}")
    print(f"  Obs space:    {model.observation_space.shape}  (expected (325,))")
    print(f"  Action space: {model.action_space.shape}  (expected (4,))")

    # ---------- Callbacks ----------
    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_synth_reweighted",
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

    # ---------- Train ----------
    print(f"\nStarting training: {TOTAL_TIMESTEPS:,} timesteps")
    print()
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    finally:
        final_model_path = ckpt_dir / "synth_reweighted_pretrain_final.zip"
        final_vecnorm_path = ckpt_dir / "synth_reweighted_pretrain_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print("\n" + "=" * 80)
    print("Reweighted synth pretrain complete.")
    print("=" * 80)
    print()
    print("Next: train/finetune_real_reweighted.py to fine-tune on real train.")


if __name__ == "__main__":
    main()