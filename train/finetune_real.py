"""
train/finetune_real.py

Fine-tune a synth-pretrained PPO policy on real train CSV.

Two presets, selected via --preset:

  light  (Option A — conservative):
    Goal: marginal refinement. Stay close to the 600k synth checkpoint
    while gently adapting to real-data feature distribution.
    LR=3e-5, n_epochs=4, clip_range=0.1, ent_coef=0.01, 200k steps.

  heavy  (Option B — growth-seeking):
    Goal: significant learning on real data, leveraging synth as warmup.
    LR=1e-4, n_epochs=4, clip_range=0.2, ent_coef=0.02, 500k steps.

Both load:
  - Model:    checkpoints/synth_pretrain/best_on_real_train/best_model.zip   (=600k)
  - Vecnorm:  checkpoints/synth_pretrain/ppo_synth_vecnormalize_600000_steps.pkl

Both train on real TRAIN CSV with periodic eval on real TEST CSV.

Run from project root:
    python train/finetune_real.py --preset light
    python train/finetune_real.py --preset heavy
"""

import argparse
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
    LowLevelPortfolioEnv,
    PortfolioCore,
    process_raw_df,
)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "light": dict(
        learning_rate=3e-5,
        n_epochs=4,
        clip_range=0.1,
        ent_coef=0.01,
        total_timesteps=200_000,
        eval_freq=50_000,
        ckpt_freq=50_000,
        n_envs=8,
        n_steps=512,
        batch_size=128,
        tag="light",
    ),
    "heavy": dict(
        learning_rate=1e-4,
        n_epochs=4,
        clip_range=0.2,
        ent_coef=0.02,
        total_timesteps=500_000,
        eval_freq=100_000,
        ckpt_freq=100_000,
        n_envs=8,
        n_steps=512,
        batch_size=128,
        tag="heavy",
    ),
}

# Source checkpoint for both fine-tunes
SOURCE_CKPT_DIR = PATHS.checkpoints / "synth_pretrain"
SOURCE_MODEL = SOURCE_CKPT_DIR / "best_on_real_train" / "best_model.zip"
SOURCE_VECNORM = SOURCE_CKPT_DIR / "ppo_synth_vecnormalize_600000_steps.pkl"

REAL_EPISODE_LENGTH = 384
N_EVAL_EPISODES = 10


# ---------------------------------------------------------------------------
# Env builders
# ---------------------------------------------------------------------------

def build_train_vec(features, returns, n_envs, source_vecnorm_path):
    """Build the training vec env, loading the saved VecNormalize stats from
    the synth pretrain so the running mean/var transitions continuously."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _make(rank):
        def _init():
            core = PortfolioCore(features, returns, cfg=cfg,
                                  rng=np.random.default_rng(2000 + rank))
            return LowLevelPortfolioEnv(core)
        return _init

    fns = [_make(i) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    vec.seed(0)

    # Load saved stats. Critical: training=True so stats keep updating during
    # fine-tune to drift toward real-data distribution; norm_reward=False to
    # match synth pretrain.
    vec = VecNormalize.load(str(source_vecnorm_path), vec)
    vec.training = True
    vec.norm_reward = False
    return vec


def build_eval_vec(features, returns, source_vecnorm_path):
    """Eval env wrapped with frozen VecNormalize stats (from the source)."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        return LowLevelPortfolioEnv(core)

    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(str(source_vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


class SyncVecNormalizeCallback(EvalCallback):
    """Sync training VecNormalize stats into eval env before each evaluation.
    Same pattern as in synth_pretrain.py."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True, choices=list(PRESETS.keys()),
                        help="Which fine-tune preset to run")
    args = parser.parse_args()
    cfg_p = PRESETS[args.preset]

    print("=" * 80)
    print(f"Fine-tune: preset = {args.preset}")
    print("=" * 80)
    print(f"  learning_rate:   {cfg_p['learning_rate']}")
    print(f"  n_epochs:        {cfg_p['n_epochs']}")
    print(f"  clip_range:      {cfg_p['clip_range']}")
    print(f"  ent_coef:        {cfg_p['ent_coef']}")
    print(f"  total_timesteps: {cfg_p['total_timesteps']:,}")
    print(f"  n_envs:          {cfg_p['n_envs']}")
    print(f"  source model:    {SOURCE_MODEL.name}")
    print(f"  source vecnorm:  {SOURCE_VECNORM.name}")

    # ---------- Verify source files ----------
    if not SOURCE_MODEL.exists():
        print(f"\nERROR: source model not found at {SOURCE_MODEL}")
        sys.exit(1)
    if not SOURCE_VECNORM.exists():
        print(f"\nERROR: source vecnorm not found at {SOURCE_VECNORM}")
        sys.exit(1)

    # ---------- Load real CSVs ----------
    print("\nLoading real CSVs...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    # ---------- Build envs ----------
    print(f"\nBuilding training vec env ({cfg_p['n_envs']} workers on real train)...")
    train_vec = build_train_vec(feats_train, rets_train,
                                  cfg_p["n_envs"], SOURCE_VECNORM)

    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec(feats_train, rets_train, SOURCE_VECNORM)
    eval_test_vec = build_eval_vec(feats_test, rets_test, SOURCE_VECNORM)

    # ---------- Output paths ----------
    ckpt_dir = PATHS.checkpoints / f"finetune_{cfg_p['tag']}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / f"finetune_{cfg_p['tag']}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    # ---------- Load model and override hyperparameters ----------
    print(f"\nLoading PPO from {SOURCE_MODEL.name} and overriding hyperparams...")
    model = PPO.load(
        str(SOURCE_MODEL),
        env=train_vec,
        device="cpu",
        # Override hyperparams from the loaded model with fine-tune values
        custom_objects={
            "learning_rate": cfg_p["learning_rate"],
            "lr_schedule": lambda _: cfg_p["learning_rate"],  # PPO needs schedule
            "clip_range": cfg_p["clip_range"],
            "clip_range_vf": None,
        },
    )
    # These get loaded from the model file but we want to override them
    model.learning_rate = cfg_p["learning_rate"]
    model.lr_schedule = lambda _: cfg_p["learning_rate"]
    model.n_epochs = cfg_p["n_epochs"]
    model.ent_coef = cfg_p["ent_coef"]
    model.clip_range = lambda _: cfg_p["clip_range"]
    model.tensorboard_log = str(tb_dir)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Loaded policy params: {n_params:,}")
    print(f"  Obs space:    {model.observation_space.shape}")
    print(f"  Action space: {model.action_space.shape}")
    print(f"  Device:       {model.device}")

    # ---------- Callbacks ----------
    eval_freq_pw = max(cfg_p["eval_freq"] // cfg_p["n_envs"], 1)
    ckpt_freq_pw = max(cfg_p["ckpt_freq"] // cfg_p["n_envs"], 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix=f"ppo_finetune_{cfg_p['tag']}",
        save_vecnormalize=True,
    )

    eval_train_cb = SyncVecNormalizeCallback(
        eval_env=eval_train_vec,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq_pw,
        log_path=str(tb_dir / "eval_real_train"),
        best_model_save_path=str(ckpt_dir / "best_on_real_train"),
        deterministic=True,
        render=False,
        verbose=1,
    )

    eval_test_cb = SyncVecNormalizeCallback(
        eval_env=eval_test_vec,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq_pw,
        log_path=str(tb_dir / "eval_real_test"),
        best_model_save_path=str(ckpt_dir / "best_on_real_test"),
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_train_cb, eval_test_cb])

    # ---------- Train ----------
    print(f"\nStarting fine-tune: {cfg_p['total_timesteps']:,} timesteps")
    print(f"  Checkpoint every {cfg_p['ckpt_freq']:,} steps "
          f"({ckpt_freq_pw} per worker)")
    print(f"  Eval every {cfg_p['eval_freq']:,} steps "
          f"({eval_freq_pw} per worker), "
          f"{N_EVAL_EPISODES} episodes each")
    print()

    try:
        model.learn(
            total_timesteps=cfg_p["total_timesteps"],
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True,  # fresh counter for this fine-tune phase
        )
    finally:
        final_model_path = ckpt_dir / f"finetune_{cfg_p['tag']}_final.zip"
        final_vecnorm_path = ckpt_dir / f"finetune_{cfg_p['tag']}_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")

        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print(f"Fine-tune ({args.preset}) complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()