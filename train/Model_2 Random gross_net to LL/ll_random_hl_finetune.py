"""
train/ll_random_hl_finetune.py

Fine-tune the LL (random-HL) synth pretrain checkpoint on real train CSV.
Mirrors the original light fine-tune but uses LowLevelPortfolioEnvRandomHL
so the LL keeps seeing varied HL actions during fine-tune. If we used fixed
HL action here, we'd undo the random-HL responsiveness the synth pretrain
just produced.

Hyperparameters identical to original light fine-tune:
    lr=3e-5, n_epochs=4, clip_range=0.1, ent_coef=0.01, 200k steps
These were validated on the original LL phase as the conservative settings
that didn't break the pretrained policy.

Source checkpoint: ll_random_hl_synth_pretrain final (1M synth steps).
Source vecnorm:    same.

Run from project root:
    python train/ll_random_hl_finetune.py
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
    PortfolioCore,
    process_raw_df,
)
from env.portfolio_hrl_env_random_hl import LowLevelPortfolioEnvRandomHL


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REAL_EPISODE_LENGTH = 384

# Source: the LL synth pretrain we just completed
SOURCE_CKPT_DIR = PATHS.checkpoints / "ll_random_hl_synth_pretrain"
SOURCE_MODEL = SOURCE_CKPT_DIR / "ll_random_hl_pretrain_final.zip"
SOURCE_VECNORM = SOURCE_CKPT_DIR / "ll_random_hl_pretrain_final_vecnorm.pkl"

N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

PPO_OVERRIDES = dict(
    learning_rate=3e-5,
    n_epochs=4,
    clip_range=0.1,
    ent_coef=0.01, # ====> 5
)


# ---------------------------------------------------------------------------
# Env builders
# ---------------------------------------------------------------------------

def build_train_vec(features, returns, source_vecnorm_path):
    """Real-train env, with HL action randomized per episode and VecNormalize
    initialized from the synth pretrain stats so the obs distribution
    transitions continuously."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def core_factory(rank: int):
        core = PortfolioCore(features, returns, cfg=cfg,
                              rng=np.random.default_rng(5000 + rank))
        hl_rng = np.random.default_rng(5000 + rank + 100_000)
        return LowLevelPortfolioEnvRandomHL(core, hl_rng=hl_rng)

    fns = [lambda i=i: core_factory(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)

    # Load saved VecNormalize stats — keep training=True so stats keep
    # updating during fine-tune (drift toward real-data distribution),
    # norm_reward=False to match synth pretrain.
    vec = VecNormalize.load(str(source_vecnorm_path), vec)
    vec.training = True
    vec.norm_reward = False
    return vec


def build_eval_vec(features, returns, source_vecnorm_path):
    """Eval env with frozen VecNormalize stats, also using random HL actions."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        hl_rng = np.random.default_rng(99999)
        return LowLevelPortfolioEnvRandomHL(core, hl_rng=hl_rng)

    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(str(source_vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
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
    print("LL random-HL fine-tune on real train")
    print("=" * 80)
    for k, v in PPO_OVERRIDES.items():
        print(f"  {k:<16s}: {v}")
    print(f"  total_timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  n_envs          : {N_TRAIN_ENVS}")
    print(f"  source model    : {SOURCE_MODEL.name}")
    print(f"  source vecnorm  : {SOURCE_VECNORM.name}")

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
    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers on real train (random HL)...")
    train_vec = build_train_vec(feats_train, rets_train, SOURCE_VECNORM)

    print("Building eval envs on real train + real test (random HL)...")
    eval_train_vec = build_eval_vec(feats_train, rets_train, SOURCE_VECNORM)
    eval_test_vec = build_eval_vec(feats_test, rets_test, SOURCE_VECNORM)

    # ---------- Output paths ----------
    ckpt_dir = PATHS.checkpoints / "ll_random_hl_finetune"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "ll_random_hl_finetune"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    # ---------- Load model and override hyperparameters ----------
    print(f"\nLoading PPO from {SOURCE_MODEL.name} and overriding hyperparams...")
    model = PPO.load(
        str(SOURCE_MODEL),
        env=train_vec,
        device="cpu",
        custom_objects={
            "learning_rate": PPO_OVERRIDES["learning_rate"],
            "lr_schedule": lambda _: PPO_OVERRIDES["learning_rate"],
            "clip_range": PPO_OVERRIDES["clip_range"],
            "clip_range_vf": None,
        },
    )
    # Belt-and-suspenders attribute overrides
    model.learning_rate = PPO_OVERRIDES["learning_rate"]
    model.lr_schedule = lambda _: PPO_OVERRIDES["learning_rate"]
    model.n_epochs = PPO_OVERRIDES["n_epochs"]
    model.ent_coef = PPO_OVERRIDES["ent_coef"]
    model.clip_range = lambda _: PPO_OVERRIDES["clip_range"]
    model.tensorboard_log = str(tb_dir)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Loaded policy params: {n_params:,}")
    print(f"  Obs space:    {model.observation_space.shape}  (expected (325,))")
    print(f"  Action space: {model.action_space.shape}  (expected (4,))")
    print(f"  Device:       {model.device}")

    # ---------- Callbacks ----------
    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_ll_random_hl_ft",
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
    print(f"\nStarting fine-tune: {TOTAL_TIMESTEPS:,} timesteps")
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} ({ckpt_freq_pw}/worker)")
    print(f"  Eval every {EVAL_FREQ:,} ({eval_freq_pw}/worker), "
          f"{N_EVAL_EPISODES} episodes each")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    finally:
        final_model_path = ckpt_dir / "ll_random_hl_ft_final.zip"
        final_vecnorm_path = ckpt_dir / "ll_random_hl_ft_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("LL random-HL fine-tune complete.")
    print("=" * 80)
    print()
    print("Next: retrain HL on this LL using train/hl_synth_pretrain.py with")
    print("the new LL paths. We expect posture/net_gap_SB_minus_Bull to actually")
    print("move this time.")


if __name__ == "__main__":
    main()