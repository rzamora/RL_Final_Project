"""
train/synth_pretrain.py

Synthetic pool pretrain — Phase 1.

Trains a PPO policy on the synthetic pool for 1M steps. Each episode samples
a fresh path from the 2000-path pool, so the agent sees genuine diversity
instead of memorizing a small number of real-CSV windows (the failure mode
of the previous 50k tiny-train).

Architecture:
  - 8 parallel envs (SubprocVecEnv) each backed by SyntheticPoolCoreSampler
  - VecNormalize on observations only (NOT reward — reward is already shaped)
  - PPO defaults from the project plan: lr=3e-4, n_steps=512, batch=128,
    gamma=0.99, clip=0.2, ent_coef=0.02 (slightly elevated for exploration)
  - LayerNormActorCriticPolicy (the project's standard)
  - CPU device (benchmarked faster than MPS for this network)

Periodic eval:
  - Every 200k steps: 10 episodes on real train CSV, 10 on real test CSV
  - Two SB3 EvalCallbacks running off frozen snapshots of VecNormalize stats
  - Logged separately to TensorBoard under eval/train and eval/test

Checkpointing:
  - Every 100k steps to checkpoints/synth_pretrain/
  - Final model + final VecNormalize saved as pretrain_final.zip / .pkl

Run from project root:
    python train/synth_pretrain.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "env"))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from project_config import PATHS
from portfolio_hrl_env_fixed import (
    CoreConfig,
    LayerNormActorCriticPolicy,
    LowLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    make_vec_env,
    process_raw_df,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Synth episode length: synth paths are 384 rows; PortfolioCore requires
# full_n_steps >= episode_length + 1, so we cap at 383 for synth.
SYNTH_EPISODE_LENGTH = 383
# Real-data evaluation uses the standard 384-step episodes.
REAL_EPISODE_LENGTH = 384

N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000 # 1_000_000
CHECKPOINT_FREQ = 100_000
EVAL_FREQ = 200_000  # in env steps per parallel worker #200_000
N_EVAL_EPISODES = 10

PPO_KWARGS = dict(
    learning_rate=1e-4,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    n_epochs=10,
    device="cpu",
    seed=0,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def build_train_vec_env(pool: dict) -> VecNormalize:
    """8 SubprocVecEnv workers, each with its own SyntheticPoolCoreSampler.
    VecNormalize on observations only — reward is already shaped."""
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def core_factory(rank: int):
        # Each worker's sampler has its own RNG so paths diverge across workers
        return SyntheticPoolCoreSampler(
            pool=pool,
            cfg=cfg,
            rng=np.random.default_rng(1000 + rank),
        )

    vec = make_vec_env(
        core_factory=core_factory,
        env_class=LowLevelPortfolioEnv,
        n_envs=N_TRAIN_ENVS,
        use_subproc=True,
        vecnormalize=False,  # we wrap manually below to control settings
        seed=0,
    )
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def build_eval_vec_env(features: np.ndarray, returns: np.ndarray, name: str) -> VecNormalize:
    """Single-env eval wrapper for a fixed real-data array. VecNormalize is
    initialized empty here; we sync stats from the training env at eval time."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _make():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        return LowLevelPortfolioEnv(core)

    vec = DummyVecEnv([_make])
    # IMPORTANT: training=False so eval doesn't update the running stats.
    # norm_reward must match the training wrapper for consistency.
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
                       training=False)
    return vec


# ---------------------------------------------------------------------------
# Sync helper for VecNormalize stats
# ---------------------------------------------------------------------------

class SyncVecNormalizeCallback(EvalCallback):
    """EvalCallback that copies the training env's VecNormalize stats into
    the eval env right before each evaluation. Without this, the eval env
    normalizes obs with stale (possibly zero-init) stats and the policy
    sees garbage."""

    def _on_step(self) -> bool:
        # Sync stats just before EvalCallback runs its evaluation
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
    print("=" * 70)
    print("Synth pretrain — 1M steps")
    print("=" * 70)

    # ---------- Load synth pool ----------
    print("\nLoading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}")
    print(f"  returns:  {pool['returns'].shape}  "
          f"(std={pool['returns'].std():.4f}, should be ~0.027)")

    # Sanity gate — if the rescale didn't take, abort before wasting 30 min
    if pool["returns"].std() > 0.5:
        raise RuntimeError(
            f"Synth returns std={pool['returns'].std():.3f} is too large. "
            f"Expected fractional units (~0.027). The percent-to-fractional "
            f"rescale in load_synthetic_pool may not be applied."
        )

    # ---------- Load real CSVs for periodic eval ----------
    print("\nLoading real CSVs for periodic eval...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train CSV: {feats_train.shape}")
    print(f"  test CSV:  {feats_test.shape}")

    # ---------- Build envs ----------
    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers on synth pool...")
    train_vec = build_train_vec_env(pool)

    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec_env(feats_train, rets_train, "real_train")
    eval_test_vec = build_eval_vec_env(feats_test, rets_test, "real_test")

    # ---------- Output paths ----------
    ckpt_dir = PATHS.checkpoints / "synth_pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "synth_pretrain"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    # ---------- Callbacks ----------
    # EvalCallback's eval_freq is in *per-worker* steps. With 8 workers and
    # EVAL_FREQ=200k total steps, the per-worker freq is 200_000 // 8 = 25_000.
    eval_freq_per_worker = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_per_worker = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_per_worker,
        save_path=str(ckpt_dir),
        name_prefix="ppo_synth",
        save_vecnormalize=True,
    )

    eval_train_cb = SyncVecNormalizeCallback(
        eval_env=eval_train_vec,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq_per_worker,
        log_path=str(tb_dir / "eval_real_train"),
        best_model_save_path=str(ckpt_dir / "best_on_real_train"),
        deterministic=True,
        render=False,
        verbose=1,
    )

    eval_test_cb = SyncVecNormalizeCallback(
        eval_env=eval_test_vec,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq_per_worker,
        log_path=str(tb_dir / "eval_real_test"),
        best_model_save_path=str(ckpt_dir / "best_on_real_test"),
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_train_cb, eval_test_cb])

    # ---------- Build model ----------
    print(f"\nBuilding PPO with LayerNormActorCriticPolicy...")
    model = PPO(
        LayerNormActorCriticPolicy,
        train_vec,
        tensorboard_log=str(tb_dir),
        **PPO_KWARGS,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}")
    print(f"  Obs space:     {model.observation_space.shape}")
    print(f"  Action space:  {model.action_space.shape}")
    print(f"  Device:        {model.device}")

    # ---------- Train ----------
    print(f"\nStarting training: {TOTAL_TIMESTEPS:,} timesteps")
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} steps "
          f"({ckpt_freq_per_worker} per worker)")
    print(f"  Eval every {EVAL_FREQ:,} steps "
          f"({eval_freq_per_worker} per worker), "
          f"{N_EVAL_EPISODES} episodes each on real train + real test")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    finally:
        # Save final state regardless of success/interruption
        final_model_path = ckpt_dir / "pretrain_final.zip"
        final_vecnorm_path = ckpt_dir / "pretrain_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")

        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 70)
    print("Synth pretrain complete.")
    print("=" * 70)
    print()
    print("Next: run evaluation script to compare median equity vs random "
          "baseline on train, test, and held-out synth paths.")


if __name__ == "__main__":
    main()