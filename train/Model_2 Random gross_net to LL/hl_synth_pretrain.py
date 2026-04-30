"""
train/hl_synth_pretrain.py

HL controller pretrain on synthetic pool, with frozen LL (light_100k).

The HL chooses [gross_signal, net_signal] ∈ [-1, +1]² each step. The frozen
LL chooses per-asset weights given the HL action. Same reward function as LL
phase; the HL learns regime-conditional posture over the LL's allocation.

Hyperparameters mirror light fine-tune of LL phase: lr=3e-5, n_epochs=4,
clip=0.1, ent_coef=0.01. These were stable for LL fine-tune and the HL is a
similar refinement task (small action space, must coexist with frozen LL).

Custom callbacks:
  - PerRegimePostureCallback: every N steps, runs one deterministic episode
    on real test and logs per-regime mean gross and mean net to TB. This is
    the diagnostic the LL phase was missing — we want to see during training
    whether the HL is actually learning regime-conditional behavior.

Run from project root:
    python train/hl_synth_pretrain.py
"""

import sys
from pathlib import Path
import gymnasium as gym

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
    HighLevelPortfolioEnv,
    LayerNormActorCriticPolicy,
    LowLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
    process_raw_df,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LL_MODEL_PATH = PATHS.checkpoints / "ll_random_hl_finetune" / "ll_random_hl_ft_final.zip"
LL_VECNORM_PATH = PATHS.checkpoints / "ll_random_hl_finetune" / "ll_random_hl_ft_final_vecnorm.pkl"

# Synth pool path-length constraint: must be < 384
SYNTH_EPISODE_LENGTH = 383
REAL_EPISODE_LENGTH = 384

N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000
EVAL_FREQ = 50_000
POSTURE_EVAL_FREQ = 50_000  # same cadence as eval; cheap (1 episode)
N_EVAL_EPISODES = 10

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
# Frozen LL helper: load model + create a VecNormalize-aware predict() function
# ---------------------------------------------------------------------------

def load_frozen_ll():
    """Loads the LL model and its VecNormalize stats, returning a callable
    that takes a 325-dim LL observation and returns the LL's deterministic
    action.

    Patches the saved VecNormalize so the last 2 dims (HL action slots) are
    NOT normalized — they were trained with constant [0.33, 0.5] and would
    otherwise force any other HL action wildly out of distribution. See
    "Fix A" in design notes.
    """
    print(f"  Loading frozen LL model: {LL_MODEL_PATH.name}")
    ll_model = PPO.load(str(LL_MODEL_PATH), device="cpu")

    # Build a single dummy env to attach VecNormalize to. We never step it;
    # we only use its obs_rms / ret_rms for normalizing observations.
    print(f"  Loading LL VecNormalize: {LL_VECNORM_PATH.name}")
    dummy_env = DummyVecEnv([lambda: _DummyObsEnv(ll_model.observation_space.shape[0])])
    ll_vecnorm = VecNormalize.load(str(LL_VECNORM_PATH), dummy_env)
    ll_vecnorm.training = False

    # Patch action-slot normalization to identity (mean=0, var=1)
    obs_dim = ll_vecnorm.obs_rms.mean.shape[0]
    n_action_slots = 2  # gross_signal, net_signal appended
    print(f"  Patching LL VecNormalize: zeroing mean/unit var on last "
          f"{n_action_slots} obs dims (HL action slots)")
    ll_vecnorm.obs_rms.mean[-n_action_slots:] = 0.0
    ll_vecnorm.obs_rms.var[-n_action_slots:] = 1.0

    def predict_fn(ll_obs_unnormalized: np.ndarray):
        """Takes a single 325-dim LL obs, normalizes with patched stats,
        returns deterministic action."""
        # VecNormalize.normalize_obs expects a batch dim
        obs_batch = ll_obs_unnormalized.reshape(1, -1).astype(np.float32)
        normalized = ll_vecnorm.normalize_obs(obs_batch)
        action, _ = ll_model.predict(normalized, deterministic=True)
        return action[0]

    return predict_fn


class _DummyObsEnv(gym.Env):
    """Minimal stub used only for VecNormalize.load — never stepped.
    Must subclass gym.Env to satisfy SB3's _patch_env validation."""
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


# ---------------------------------------------------------------------------
# A LL wrapper that mimics the SB3 PPO .predict() signature, used by HighLevelPortfolioEnv
# ---------------------------------------------------------------------------

class FrozenLLAdapter:
    """HighLevelPortfolioEnv expects ll_model.predict(obs, deterministic=...).
    This adapter wraps the patched-VecNormalize predict_fn to match that
    signature."""
    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def predict(self, obs, deterministic=True):
        # Always deterministic regardless of flag — this is a frozen policy
        action = self._predict_fn(obs)
        return action, None


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def build_train_vec(pool, ll_adapter):
    cfg = CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)

    def _make(rank):
        def _init():
            sampler = SyntheticPoolCoreSampler(
                pool=pool, cfg=cfg,
                rng=np.random.default_rng(3000 + rank),
            )
            return HighLevelPortfolioEnv(sampler, ll_adapter)
        return _init

    fns = [_make(i) for i in range(N_TRAIN_ENVS)]
    vec = SubprocVecEnv(fns) if N_TRAIN_ENVS > 1 else DummyVecEnv(fns)
    vec.seed(0)

    # Fresh VecNormalize for HL — the HL obs space (323) differs from LL (325),
    # so we cannot reuse LL's stats. Train from scratch on HL obs distribution.
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
    """Every eval_freq steps (per-worker), runs ONE deterministic episode
    on real test from day 0 (fixed window) and logs per-regime mean gross
    and mean net to TensorBoard.

    This is the central diagnostic for HL training: are we learning
    regime-conditional posture?"""

    def __init__(self, features, returns, regime_idx, ll_adapter,
                 eval_freq, train_vec_for_norm, verbose=0):
        super().__init__(verbose)
        self.features = features
        self.returns = returns
        self.regime_idx = regime_idx  # (T,) array of regime labels per day
        self.ll_adapter = ll_adapter
        self.eval_freq = eval_freq
        self.train_vec_for_norm = train_vec_for_norm
        self.regime_names = ["Bull", "Bear", "SevereBear", "Crisis"]

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Build a fresh fixed-window env each time
        cfg = CoreConfig(episode_length=REAL_EPISODE_LENGTH)
        core = PortfolioCore(self.features, self.returns, cfg=cfg,
                              rng=np.random.default_rng(0))
        # Pin to day 0 for reproducible posture diagnostic
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
            # Normalize HL obs using current training VecNormalize stats
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

        # Per-regime stats
        T = len(gross_arr)
        regime_slice = self.regime_idx[:T]  # align

        for i, name in enumerate(self.regime_names):
            mask = regime_slice == i
            if mask.sum() > 0:
                self.logger.record(f"posture/gross_{name}", float(gross_arr[mask].mean()))
                self.logger.record(f"posture/net_{name}",   float(net_arr[mask].mean()))
                self.logger.record(f"posture/n_{name}",     int(mask.sum()))

        # Overall episode stats
        self.logger.record("posture/gross_overall_mean", float(gross_arr.mean()))
        self.logger.record("posture/net_overall_mean",   float(net_arr.mean()))
        self.logger.record("posture/gross_std",          float(gross_arr.std()))
        self.logger.record("posture/net_std",            float(net_arr.std()))

        # The headline diagnostic: gap between SevereBear and Bull
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
    print("HL synth pretrain — 1M steps")
    print("=" * 80)

    # ---------- Verify LL files exist ----------
    if not LL_MODEL_PATH.exists():
        print(f"\nERROR: LL model not found: {LL_MODEL_PATH}")
        sys.exit(1)
    if not LL_VECNORM_PATH.exists():
        print(f"\nERROR: LL vecnorm not found: {LL_VECNORM_PATH}")
        sys.exit(1)

    # ---------- Load synth pool ----------
    print("\nLoading synthetic pool...")
    pool = load_synthetic_pool(PATHS.synth_pool)
    print(f"  features: {pool['features'].shape}  returns std: {pool['returns'].std():.4f}")
    if pool["returns"].std() > 0.5:
        raise RuntimeError("Synth returns not in fractional units — "
                           "load_synthetic_pool patch may not be applied.")

    # ---------- Load real CSVs for periodic eval ----------
    print("\nLoading real CSVs...")
    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    feats_train, rets_train, _ = process_raw_df(train_df)
    feats_test, rets_test, _ = process_raw_df(test_df)
    print(f"  train: {feats_train.shape}, test: {feats_test.shape}")

    # Pre-compute test regime labels (argmax of the four regime_prob_* cols)
    regime_cols = ["regime_prob_Bull", "regime_prob_Bear",
                   "regime_prob_SevereBear", "regime_prob_Crisis"]
    test_regime_idx = test_df[regime_cols].to_numpy().argmax(axis=1)
    print(f"  test regime distribution: "
          f"Bull={(test_regime_idx==0).sum()}  "
          f"Bear={(test_regime_idx==1).sum()}  "
          f"SevereBear={(test_regime_idx==2).sum()}  "
          f"Crisis={(test_regime_idx==3).sum()}")

    # ---------- Load frozen LL ----------
    print("\nLoading frozen LL...")
    ll_predict_fn = load_frozen_ll()
    ll_adapter = FrozenLLAdapter(ll_predict_fn)

    # Quick sanity check: invoke the LL on a dummy obs to confirm it runs
    dummy_obs = np.zeros(325, dtype=np.float32)
    _ = ll_predict_fn(dummy_obs)
    print(f"  Frozen LL: predict() works on a dummy 325-dim obs")

    # ---------- Build envs ----------
    print(f"\nBuilding {N_TRAIN_ENVS} SubprocVecEnv workers on synth pool...")
    train_vec = build_train_vec(pool, ll_adapter)

    print("Building eval envs on real train + real test...")
    eval_train_vec = build_eval_vec_real(feats_train, rets_train, ll_adapter)
    eval_test_vec = build_eval_vec_real(feats_test, rets_test, ll_adapter)

    # ---------- Output paths ----------
    ckpt_dir = PATHS.checkpoints / "hl_synth_pretrain_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = PATHS.tb_logs / "hl_synth_pretrain_v2"
    tb_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints -> {ckpt_dir}")
    print(f"TensorBoard -> {tb_dir}")

    # ---------- Build PPO model (fresh, not loaded) ----------
    print("\nBuilding HL PPO with LayerNormActorCriticPolicy (fresh init)...")
    model = PPO(
        LayerNormActorCriticPolicy,
        train_vec,
        tensorboard_log=str(tb_dir),
        **PPO_KWARGS,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  HL policy params: {n_params:,}")
    print(f"  HL obs space:     {model.observation_space.shape}  (expected (323,))")
    print(f"  HL action space:  {model.action_space.shape}  (expected (2,))")

    # ---------- Callbacks ----------
    eval_freq_pw = max(EVAL_FREQ // N_TRAIN_ENVS, 1)
    ckpt_freq_pw = max(CHECKPOINT_FREQ // N_TRAIN_ENVS, 1)
    posture_freq_pw = max(POSTURE_EVAL_FREQ // N_TRAIN_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq_pw,
        save_path=str(ckpt_dir),
        name_prefix="ppo_hl",
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

    # ---------- Train ----------
    print(f"\nStarting HL training: {TOTAL_TIMESTEPS:,} timesteps")
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
        final_model_path = ckpt_dir / "hl_pretrain_final.zip"
        final_vecnorm_path = ckpt_dir / "hl_pretrain_final_vecnorm.pkl"
        model.save(final_model_path)
        train_vec.save(str(final_vecnorm_path))
        print(f"\nFinal model -> {final_model_path}")
        print(f"Final vecnorm -> {final_vecnorm_path}")
        train_vec.close()
        eval_train_vec.close()
        eval_test_vec.close()

    print()
    print("=" * 80)
    print("HL synth pretrain complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()