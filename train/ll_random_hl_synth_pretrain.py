"""
train/ll_random_hl_synth_pretrain.py

LL synth pretrain (1M steps) with HL action randomized per episode.

Why this exists: the original LL synth pretrain trained with a fixed HL
action [0.33, 0.5]. When the resulting LL was later frozen and wrapped by
the HL, the LL's response to off-distribution HL actions broke HL training
(HL never learned regime-conditional posture; net_gap_SB_minus_Bull stayed
within ±0.02 across 1M steps).

This rerun trains the LL to be robust across the full HL action space by
sampling HL action ~ Uniform(-1, +1)^2 every episode. Output of this run +
subsequent fine-tune will be wrapped by retrained HL training.

Hyperparameter changes from original LL synth pretrain:
  - n_epochs: 10 -> 4. The original run had KL > 1.0 by 800k steps because
    n_epochs=10 over-amplified each rollout's gradient direction. n_epochs=4
    is the standard recipe for taming KL when LR alone isn't enough.
  - All other PPO hyperparameters identical.

Run from project root:
    python train/ll_random_hl_synth_pretrain.py
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
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)
from env.portfolio_hrl_env_random_hl import LowLevelPortfolioEnvRandomHL


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
    n_epochs=4,                 # was 10 in original; cuts KL inflation
    device="cpu",
    seed=0,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def build_train_vec(pool):
    """8 SubprocVecEnv workers on synth pool, LL with randomized HL action.

    Each worker has two distinct RNGs: one for the core (synth path sampler)
    and one for the HL action randomizer. Different ranks get different seeds
    on both, ensuring full diversity across workers.
    """
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def core_factory(rank: int):
        # Each worker's sampler has its own RNG
        sampler = SyntheticPoolCoreSampler(
            pool=pool, cfg=cfg,
            rng=np.random.default_rng(4000 + rank),
        )
        # And its own HL action RNG, offset to decorrelate from core RNG
        hl_rng = np.random.default_rng(4000 + rank + 100_000)
        env = LowLevelPortfolioEnvRandomHL(sampler, hl_rng=hl_rng)
        return env

    fns = [lambda i=i: core_factory(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def build_eval_vec(features, returns):
    """Eval env on real CSV. HL action still randomized (same as training
    distribution) so eval reflects what the LL will see in deployment under
    the HL. VecNormalize loaded from training env at eval time."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        # Fixed HL RNG for reproducibility within eval
        hl_rng = np.random.default_rng(99999)
        return LowLevelPortfolioEnvRandomHL(core, hl_rng=hl_rng)

    vec = DummyVecEnv([_init])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
                       training=False)
    return vec


# ---------------------------------------------------------------------------
# Sync VecNormalize stats into eval env at eval time
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
    print("LL synth pretrain with random HL — 1M steps")
    print("=" * 80)
    print()
    print("Key difference from original LL synth pretrain:")
    print("  HL action sampled Uniform(-1, +1)^2 per episode (not fixed at [0.33, 0.5])")
    print("  PPO n_epochs=4 (was 10)")
    print()

    # ---------- Load synth pool ----------
    print("Loading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}")
    print(f"  returns:  {pool['returns'].shape}  std={pool['returns'].std():.4f}")
    if pool["returns"].std() > 0.5:
        raise RuntimeError(
            f"Synth returns std={pool['returns'].std():.3f} too large. "
            f"Expected fractional units (~0.027). Patch in load_synthetic_pool "
            f"may not be applied."
        )

    # ---------- Load real CSVs for periodic eval ----------
    print("\nLoading real CSVs for periodic eval...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    # ---------- Build envs ----------
    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers (synth pool, random HL)...")
    train_vec = build_train_vec(pool)

    print("Building eval envs on real train + real test (random HL too)...")
    eval_train_vec = build_eval_vec(feats_train, rets_train)
    eval_test_vec = build_eval_vec(feats_test, rets_test)

    # ---------- Output paths ----------
    ckpt_dir = PATHS.checkpoints / "ll_random_hl_synth_pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "ll_random_hl_synth_pretrain"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    # ---------- PPO model ----------
    print(f"\nBuilding PPO with LayerNormActorCriticPolicy...")
    model = PPO(
        LayerNormActorCriticPolicy,
        train_vec,
        tensorboard_log=str(tb_dir),
        **PPO_KWARGS,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}")
    print(f"  Obs space:     {model.observation_space.shape}  (expected (325,))")
    print(f"  Action space:  {model.action_space.shape}  (expected (4,))")
    print(f"  Device:        {model.device}")

    # ---------- Callbacks ----------
    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_ll_random_hl",
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
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} ({ckpt_freq_pw}/worker)")
    print(f"  Eval every {EVAL_FREQ:,} ({eval_freq_pw}/worker), "
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
        final_model_path = ckpt_dir / "ll_random_hl_pretrain_final.zip"
        final_vecnorm_path = ckpt_dir / "ll_random_hl_pretrain_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("LL random-HL synth pretrain complete.")
    print("=" * 80)
    print()
    print("Next: ll_random_hl_finetune.py to fine-tune this checkpoint on real train.")


if __name__ == "__main__":
    main()