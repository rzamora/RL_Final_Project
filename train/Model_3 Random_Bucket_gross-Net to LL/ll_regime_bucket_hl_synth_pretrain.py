"""
train/ll_regime_bucket_hl_synth_pretrain.py

LL synth pretrain (1M steps) with regime-bucketed HL action sampling.

Why this exists: v2 (LL trained with Uniform[-1,+1]^2 HL actions) gave the LL
to handle any HL action gracefully, but when wrapped under HL training the HL
still failed to learn regime-conditional posture (net_gap_SB_minus_Bull
~0 throughout). Hypothesis: random uniform HL training spent capacity teaching
the LL to handle objectively-wrong combinations (e.g. Bull regime + gross=-1
+ net=-1) which never appear at deployment, diluting LL quality on the
combinations that matter.

This rerun trains the LL only on regime-appropriate HL actions:
  Crisis     -> bucket [-1.0, -0.5] x [-1.0, -0.5]
  SevereBear -> bucket [-0.5,  0.0] x [-0.5,  0.0]
  Bear       -> bucket [ 0.0, +0.5] x [ 0.0, +0.5]
  Bull       -> bucket [+0.5, +1.0] x [+0.5, +1.0]

Buckets are disjoint and together span the diagonal of [-1,+1]^2. Modal
regime per episode determines the bucket; HL action is sampled uniformly
within the bucket and held constant for the episode.

Hyperparameters identical to ll_random_hl_synth_pretrain (lr=1e-4,
n_epochs=4, clip=0.2, ent_coef=0.02). Same training-time eval cadence,
same n_envs=8, same 1M step budget.

Run from project root:
    python train/ll_regime_bucket_hl_synth_pretrain.py
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
    load_synthetic_pool,
    process_raw_df,
)
from env.portfolio_hrl_env_regime_bucket import (
    LowLevelPortfolioEnvRegimeBucketHL,
    SyntheticRegimeBucketCoreSampler,
    REGIME_NAMES,
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
# Real CSV: extract regime labels from regime_prob_* columns
# ---------------------------------------------------------------------------

def extract_regime_labels(df: pd.DataFrame) -> np.ndarray:
    """Argmax over regime_prob_* columns -> integer regime label per day.
    Returns shape (n_days,) with values in {0, 1, 2, 3} corresponding to
    {Bull, Bear, SevereBear, Crisis}."""
    cols = ["regime_prob_Bull", "regime_prob_Bear",
            "regime_prob_SevereBear", "regime_prob_Crisis"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing regime probability columns: {missing}. "
            f"Available cols: {list(df.columns)[:20]}..."
        )
    return df[cols].to_numpy().argmax(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def build_train_vec(pool):
    """Synth-pool vec env with regime-bucket HL sampling. 8 SubprocVecEnv
    workers, each with independent core RNG and HL RNG."""
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def core_factory(rank: int):
        sampler = SyntheticRegimeBucketCoreSampler(
            pool=pool, cfg=cfg,
            rng=np.random.default_rng(7000 + rank),
        )
        hl_rng = np.random.default_rng(7000 + rank + 100_000)
        env = LowLevelPortfolioEnvRegimeBucketHL(sampler, hl_rng=hl_rng)
        return env

    fns = [lambda i=i: core_factory(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def build_eval_vec(features, returns, regime_labels):
    """Real-CSV eval env with regime-bucket HL sampling. Same modal-regime
    convention as train; the bucket is determined by the dominant regime
    in the eval window."""
    cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)

    def _init():
        core = PortfolioCore(features, returns, cfg=cfg, rng=np.random.default_rng(0))
        hl_rng = np.random.default_rng(99999)
        return LowLevelPortfolioEnvRegimeBucketHL(
            core,
            episode_regime_labels=regime_labels,
            hl_rng=hl_rng,
        )

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
    print("LL synth pretrain with regime-bucketed HL — 1M steps")
    print("=" * 80)
    print()
    print("HL action sampling: regime-bucketed (Option A, disjoint corners)")
    print("  Crisis     -> [-1.0, -0.5] x [-1.0, -0.5]")
    print("  SevereBear -> [-0.5,  0.0] x [-0.5,  0.0]")
    print("  Bear       -> [ 0.0, +0.5] x [ 0.0, +0.5]")
    print("  Bull       -> [+0.5, +1.0] x [+0.5, +1.0]")
    print()

    # ---------- Load synth pool with regimes ----------
    print("Loading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}")
    print(f"  returns:  {pool['returns'].shape}  std={pool['returns'].std():.4f}")
    if pool.get("regimes") is None:
        raise RuntimeError(
            "Pool dict does not contain 'regimes' key — cannot use regime-bucket env. "
            "Re-load synth pool .npz and confirm 'regimes' is present."
        )
    print(f"  regimes:  {pool['regimes'].shape}  dtype={pool['regimes'].dtype}")
    if pool["returns"].std() > 0.5:
        raise RuntimeError(
            f"Synth returns std={pool['returns'].std():.3f} too large. "
            f"Expected fractional units (~0.027)."
        )

    # Quick distribution check
    flat = pool["regimes"].flatten()
    counts = np.bincount(flat, minlength=4)
    total = counts.sum()
    print(f"  per-step regime distribution:")
    for i, name in enumerate(REGIME_NAMES):
        pct = 100.0 * counts[i] / total
        print(f"    {name:<11s} {counts[i]:>10d}  ({pct:5.1f}%)")

    # Per-path modal regime distribution — this determines the bucket
    # distribution during training
    path_modal = []
    for p in range(pool["regimes"].shape[0]):
        cnts = np.bincount(pool["regimes"][p], minlength=4)
        path_modal.append(int(np.argmax(cnts)))
    path_modal = np.array(path_modal)
    print(f"  per-path MODAL regime distribution (drives bucket sampling):")
    for i, name in enumerate(REGIME_NAMES):
        cnt = int((path_modal == i).sum())
        pct = 100.0 * cnt / len(path_modal)
        print(f"    {name:<11s} {cnt:>10d}  ({pct:5.1f}%)")

    # ---------- Load real CSVs for periodic eval ----------
    print("\nLoading real CSVs for periodic eval...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    regime_train = extract_regime_labels(train_df)
    regime_test = extract_regime_labels(test_df)
    print(f"  train: {feats_train.shape}, regimes: {regime_train.shape}")
    print(f"  test:  {feats_test.shape},  regimes: {regime_test.shape}")
    counts_test = np.bincount(regime_test, minlength=4)
    print(f"  test per-day regimes: "
          f"Bull={counts_test[0]} Bear={counts_test[1]} "
          f"SB={counts_test[2]} Crisis={counts_test[3]}")

    # ---------- Build envs ----------
    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers (synth, regime-bucket)...")
    train_vec = build_train_vec(pool)

    print("Building eval envs on real train + real test (regime-bucket)...")
    eval_train_vec = build_eval_vec(feats_train, rets_train, regime_train)
    eval_test_vec = build_eval_vec(feats_test, rets_test, regime_test)

    # ---------- Output paths ----------
    ckpt_dir = PATHS.checkpoints / "ll_regime_bucket_hl_synth_pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "ll_regime_bucket_hl_synth_pretrain"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    # ---------- Build PPO model (fresh init) ----------
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

    # ---------- Callbacks ----------
    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_ll_regime_bucket",
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
          f"{N_EVAL_EPISODES} episodes each")
    print()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    finally:
        final_model_path = ckpt_dir / "ll_regime_bucket_pretrain_final.zip"
        final_vecnorm_path = ckpt_dir / "ll_regime_bucket_pretrain_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("LL regime-bucket synth pretrain complete.")
    print("=" * 80)
    print()
    print("Next:")
    print("  1. Run ll_regime_bucket_hl_finetune.py (200k real-train fine-tune)")
    print("  2. Run per-bucket diagnostic: confirm LL responds correctly within")
    print("     each regime's bucket (and that responses differ across buckets).")
    print("  3. Retrain HL on top: hl_synth_pretrain_v3.py")


if __name__ == "__main__":
    main()