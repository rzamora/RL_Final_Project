"""
train/hl_ll_joint_synth_pretrain.py

Joint HL+LL training, 1M synth pretrain.

Single PPO model with structured policy:
  - Shared trunk (313+10+2 = 325-dim obs -> 256-dim latent)
  - HL head: latent -> 2-dim Diag Gaussian
  - LL head: [latent, HL_sample] -> 4-dim Diag Gaussian
  - Value head: latent -> scalar

Both heads update together via joint PPO objective. The hierarchy is
preserved structurally (HL output flows into LL input) but trained
end-to-end so the LL adapts to the HL's evolving policy in real time
and vice versa.

This is the principled answer to the frozen-LL coordination failure
documented in v1, v2, v3. Expected to either:
  (a) beat the LL-alone baseline (positive HRL result)
  (b) match it (HRL adds robustness without adding alpha)
  (c) underperform it (HRL is structurally wrong for this task)

Run from project root:
    python train/hl_ll_joint_synth_pretrain.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
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
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)
from env.portfolio_hrl_env_joint import JointHLLLPortfolioEnv
from policy.joint_hl_ll_policy import JointHLLLPolicy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYNTH_EPISODE_LENGTH = 383
REAL_EPISODE_LENGTH = 384

N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000
EVAL_FREQ = 50_000
POSTURE_EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10

# Hyperparameters: same as the v3 HL training (since we're now the HL+LL combined).
# lr=3e-5 was conservative-but-stable for v1/v2/v3 HL phase.
# Higher lr risks destabilizing the joint dynamics. Start safe.
PPO_KWARGS = dict(
    learning_rate=3e-5,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
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
                rng=np.random.default_rng(11000 + rank),
            )
            return JointHLLLPortfolioEnv(sampler)
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
        return JointHLLLPortfolioEnv(core)

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


class JointPostureCallback(BaseCallback):
    """Diagnostic for joint training. Same per-regime gross/net diagnostic
    as v1/v2/v3 HL callbacks. Runs deterministic episode on real test from
    day 0 and logs per-regime mean gross and mean net.
    """

    def __init__(self, features, returns, regime_idx,
                 eval_freq, train_vec_for_norm, verbose=0):
        super().__init__(verbose)
        self.features = features
        self.returns = returns
        self.regime_idx = regime_idx
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

        env = JointHLLLPortfolioEnv(core)
        obs, _ = env.reset()

        gross_history = []
        net_history = []
        done = False
        while not done:
            obs_batch = obs.reshape(1, -1).astype(np.float32)
            obs_norm = self.train_vec_for_norm.normalize_obs(obs_batch)[0]
            joint_action, _ = self.model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(joint_action)
            hl_action = joint_action[:2]
            gross, net = core.parse_hl_action(hl_action)
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
    print("Joint HL+LL training — 1M synth pretrain (Option B+B1)")
    print("=" * 80)
    print()
    print("Single PPO with structured policy:")
    print("  - Shared LayerNorm trunk (325-dim obs -> 256-dim latent)")
    print("  - HL head: latent -> 2-dim Diag Gaussian")
    print("  - LL head: [latent, HL_sample] -> 4-dim Diag Gaussian")
    print("  - Both update together; gradient flows HL <- LL via reparameterization")
    print()

    print("Loading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}  returns std: {pool['returns'].std():.4f}")

    print("\nLoading real CSVs...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    regime_cols = ["regime_prob_Bull", "regime_prob_Bear",
                   "regime_prob_SevereBear", "regime_prob_Crisis"]
    test_regime_idx = test_df[regime_cols].to_numpy().argmax(axis=1)
    print(f"  test regime distribution: "
          f"Bull={(test_regime_idx==0).sum()}  "
          f"Bear={(test_regime_idx==1).sum()}  "
          f"SevereBear={(test_regime_idx==2).sum()}  "
          f"Crisis={(test_regime_idx==3).sum()}")

    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers (joint env)...")
    train_vec = build_train_vec(pool)

    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec_real(feats_train, rets_train)
    eval_test_vec = build_eval_vec_real(feats_test, rets_test)

    ckpt_dir = PATHS.checkpoints / "hl_ll_joint_synth_pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "hl_ll_joint_synth_pretrain"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    print("\nBuilding PPO with JointHLLLPolicy...")
    model = PPO(
        JointHLLLPolicy,
        train_vec,
        tensorboard_log=str(tb_dir),
        **PPO_KWARGS,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Joint policy params: {n_params:,}")
    print(f"  Obs space:    {model.observation_space.shape}  (expected (325,))")
    print(f"  Action space: {model.action_space.shape}  (expected (6,))")

    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)
    posture_freq_pw = max(POSTURE_EVAL_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_joint",
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

    posture_cb = JointPostureCallback(
        features=feats_test, returns=rets_test,
        regime_idx=test_regime_idx,
        eval_freq=posture_freq_pw,
        train_vec_for_norm=train_vec,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_train_cb, eval_test_cb, posture_cb])

    print(f"\nStarting joint training: {TOTAL_TIMESTEPS:,} timesteps")
    print(f"  Checkpoint every {CHECKPOINT_FREQ:,} ({ckpt_freq_pw}/worker)")
    print(f"  Eval every {EVAL_FREQ:,} ({eval_freq_pw}/worker), "
          f"{N_EVAL_EPISODES} episodes")
    print(f"  Posture diagnostic every {POSTURE_EVAL_FREQ:,} "
          f"({posture_freq_pw}/worker), 1 episode on real test")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    finally:
        final_model_path = ckpt_dir / "joint_pretrain_final.zip"
        final_vecnorm_path = ckpt_dir / "joint_pretrain_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("Joint synth pretrain complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()