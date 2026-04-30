"""
eval/feature_importance.py

Permutation importance for HL or LL policies. For each feature, shuffles
its values across timesteps and measures how much the policy's actions
change vs the baseline (unshuffled) actions.

Output: ranked list of features by importance (action-space deviation).

Usage: edit MODEL_TYPE and checkpoint paths at the top, then run.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from project_config import PATHS


# ---------------------------------------------------------------------------
# CONFIG — edit these to choose which model to analyze
# ---------------------------------------------------------------------------

# Choose "HL" or "LL"
MODEL_TYPE = "HL"

# Pick a checkpoint to analyze. Examples below — uncomment one block:

# Option A: HL v2 with regime probs (the project's main HL)
MODEL_PATH = PATHS.checkpoints / "hl_finetune_real_unconstrained" / "hl_ft_uc_final.zip"
VECNORM_PATH = PATHS.checkpoints / "hl_finetune_real_unconstrained" / "hl_ft_uc_final_vecnorm.pkl"

# Option B: LL alone (light_100k)
# MODEL_PATH = PATHS.checkpoints / "ll_finetune_real" / "best_on_real_test" / "best_model.zip"
# VECNORM_PATH = PATHS.checkpoints / "ll_finetune_real" / "ppo_ll_ft_vecnormalize_100000_steps.pkl"

# Number of timesteps to analyze. More = more stable importance estimates.
# 829 = full real_test, ~30 sec per feature on CPU.
N_STEPS = 500

# Number of shuffles per feature (importance estimate stability)
N_SHUFFLES = 5

# Random seed for shuffling
SEED = 42


# ---------------------------------------------------------------------------
# Helper: load model + vecnorm
# ---------------------------------------------------------------------------

class _DummyEnv:
    """Minimal stub so VecNormalize can load."""
    def __init__(self, obs_dim, action_dim):
        from gymnasium import spaces
        import gymnasium as gym

        class _E(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(-1e6, 1e6, (obs_dim,), dtype=np.float32)
                self.action_space = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)
            def reset(self, **kw): return np.zeros(obs_dim, dtype=np.float32), {}
            def step(self, a): return np.zeros(obs_dim, dtype=np.float32), 0.0, True, False, {}
        self._cls = _E

    def make(self):
        return self._cls()


def load_policy(model_path, vecnorm_path):
    model = PPO.load(str(model_path), device="cpu")
    obs_dim = model.observation_space.shape[0]
    action_dim = model.action_space.shape[0]

    stub = _DummyEnv(obs_dim, action_dim)
    venv = DummyVecEnv([stub.make])
    vecnorm = VecNormalize.load(str(vecnorm_path), venv)
    vecnorm.training = False

    def predict_fn(obs):
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        normalized = vecnorm.normalize_obs(obs_batch)
        action, _ = model.predict(normalized, deterministic=True)
        return action[0]

    return predict_fn, obs_dim, action_dim


# ---------------------------------------------------------------------------
# Build observations from real_test
# ---------------------------------------------------------------------------

def build_test_observations(model_type):
    """Build a sequence of observations as the env would produce them on real_test."""
    if model_type == "HL":
        from env.portfolio_hrl_env_fixed import (
            CoreConfig, HighLevelPortfolioEnv, PortfolioCore, process_raw_df,
        )
        env_cls = HighLevelPortfolioEnv
        needs_ll = True
    else:  # LL
        from env.portfolio_hrl_env_fixed import (
            CoreConfig, LowLevelPortfolioEnv, PortfolioCore, process_raw_df,
        )
        env_cls = LowLevelPortfolioEnv
        needs_ll = False

    test_df = pd.read_csv(PATHS.test_csv)
    feats_test, rets_test, _ = process_raw_df(test_df)

    cfg = CoreConfig(episode_length=384)
    core = PortfolioCore(feats_test, rets_test, cfg=cfg, rng=np.random.default_rng(0))

    # Force episode to start at index 0
    core.t_start = 0
    core.t_end = min(384, len(feats_test) - 1)
    core.t = 0
    core.equity = cfg.initial_equity
    core.bench_equity = cfg.initial_equity
    core._init_windows()
    core.equity_window_short.append(core.equity)
    core.equity_window_long.append(core.equity)
    core.bench_equity_window_short.append(core.bench_equity)
    core.bench_equity_window_long.append(core.bench_equity)

    if needs_ll:
        # HL needs a frozen LL adapter. Use a no-op stub returning a fixed action.
        class StubLL:
            def predict(self, obs, deterministic=True):
                return np.zeros(4, dtype=np.float32), None
        env = env_cls(core, StubLL())
    else:
        env = env_cls(core, fixed_hl_action=np.array([0.33, 0.5], dtype=np.float32))

    obs_seq = []
    obs, _ = env.reset()
    obs_seq.append(obs.copy())
    for _ in range(min(N_STEPS, len(feats_test) - 2)):
        if hasattr(env, "step"):
            # Use deterministic action so observations are reproducible
            if model_type == "HL":
                action = np.array([0.0, 0.0], dtype=np.float32)
            else:
                action = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
            obs_seq.append(obs.copy())

    return np.array(obs_seq, dtype=np.float32)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def compute_baseline_actions(predict_fn, obs_array):
    return np.array([predict_fn(obs) for obs in obs_array])


def compute_permuted_actions(predict_fn, obs_array, feature_idx, rng):
    obs_perm = obs_array.copy()
    perm = rng.permutation(len(obs_array))
    obs_perm[:, feature_idx] = obs_array[perm, feature_idx]
    return np.array([predict_fn(obs) for obs in obs_perm])


def feature_importance(predict_fn, obs_array, n_features, n_shuffles, seed):
    print(f"  Computing baseline actions on {len(obs_array)} obs...")
    baseline_actions = compute_baseline_actions(predict_fn, obs_array)

    rng = np.random.default_rng(seed)
    importance = np.zeros(n_features)

    print(f"  Computing per-feature importance ({n_features} features × {n_shuffles} shuffles)...")
    for i in range(n_features):
        if i % 50 == 0:
            print(f"    feature {i}/{n_features}")
        diffs = []
        for s in range(n_shuffles):
            permuted = compute_permuted_actions(predict_fn, obs_array, i, rng)
            diff = np.mean((permuted - baseline_actions) ** 2)
            diffs.append(diff)
        importance[i] = np.mean(diffs)

    return importance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print(f"Feature importance: {MODEL_TYPE} policy")
    print("=" * 80)
    print(f"\nModel:    {MODEL_PATH.name}")
    print(f"VecNorm:  {VECNORM_PATH.name}")

    if not MODEL_PATH.exists():
        print(f"\nERROR: model not found: {MODEL_PATH}")
        sys.exit(1)
    if not VECNORM_PATH.exists():
        print(f"\nERROR: vecnorm not found: {VECNORM_PATH}")
        sys.exit(1)

    print("\nLoading model...")
    predict_fn, obs_dim, action_dim = load_policy(MODEL_PATH, VECNORM_PATH)
    print(f"  obs_dim:    {obs_dim}")
    print(f"  action_dim: {action_dim}")

    print("\nBuilding observation sequence on real_test...")
    obs_array = build_test_observations(MODEL_TYPE)
    print(f"  shape: {obs_array.shape}")

    if obs_array.shape[0] < 50:
        print("WARNING: very few observations. Importance estimates will be unstable.")

    print(f"\nComputing permutation importance ({N_SHUFFLES} shuffles per feature)...")
    importance = feature_importance(predict_fn, obs_array, obs_dim, N_SHUFFLES, SEED)

    # Build feature names
    train_df = pd.read_csv(PATHS.train_csv)
    feature_cols = [c for c in train_df.columns
                    if c not in ["date", "NVDA_close", "AMD_close", "SMH_close", "TLT_close"]]
    portfolio_state_names = ["equity", "excess_dd_short", "gross", "net",
                              "w_NVDA", "w_AMD", "w_SMH", "w_TLT",
                              "quarterly_excess", "quarterly_bench"]
    if MODEL_TYPE == "HL":
        feature_names = feature_cols + portfolio_state_names
    else:  # LL
        feature_names = feature_cols + portfolio_state_names + ["hl_gross_signal", "hl_net_signal"]

    if len(feature_names) != obs_dim:
        print(f"WARNING: name count {len(feature_names)} != obs_dim {obs_dim}. Using indices.")
        feature_names = [f"feat_{i}" for i in range(obs_dim)]

    # Print top 30 most important features
    print("\n" + "=" * 80)
    print(f"TOP 30 MOST IMPORTANT FEATURES")
    print("=" * 80)
    print(f"  {'rank':>4s}  {'idx':>4s}  {'feature':<35s}  {'importance':>12s}  {'pct':>7s}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*35}  {'-'*12}  {'-'*7}")
    sorted_idx = np.argsort(-importance)
    total_importance = importance.sum()
    for rank, i in enumerate(sorted_idx[:30]):
        pct = 100.0 * importance[i] / total_importance if total_importance > 0 else 0.0
        print(f"  {rank+1:>4d}  {i:>4d}  {feature_names[i]:<35s}  {importance[i]:>12.6f}  {pct:>6.2f}%")

    # Print regime_prob_* importance specifically
    print("\n" + "=" * 80)
    print("REGIME PROB FEATURES")
    print("=" * 80)
    regime_indices = [i for i, n in enumerate(feature_names) if "regime_prob" in n.lower()]
    if regime_indices:
        print(f"  {'idx':>4s}  {'feature':<35s}  {'importance':>12s}  {'rank':>6s}  {'pct':>7s}")
        print(f"  {'-'*4}  {'-'*35}  {'-'*12}  {'-'*6}  {'-'*7}")
        for i in regime_indices:
            rank = int(np.where(sorted_idx == i)[0][0]) + 1
            pct = 100.0 * importance[i] / total_importance if total_importance > 0 else 0.0
            print(f"  {i:>4d}  {feature_names[i]:<35s}  {importance[i]:>12.6f}  {rank:>6d}  {pct:>6.2f}%")
    else:
        print("  (no regime_prob features found in this observation)")

    # Print bottom 10 (effectively ignored features)
    print("\n" + "=" * 80)
    print("BOTTOM 10 FEATURES (effectively ignored by policy)")
    print("=" * 80)
    print(f"  {'rank':>4s}  {'idx':>4s}  {'feature':<35s}  {'importance':>12s}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*35}  {'-'*12}")
    for rank, i in enumerate(sorted_idx[-10:]):
        actual_rank = obs_dim - 9 + rank
        print(f"  {actual_rank:>4d}  {i:>4d}  {feature_names[i]:<35s}  {importance[i]:>12.6f}")

    # Save full results
    results_df = pd.DataFrame({
        "feature_idx": list(range(obs_dim)),
        "feature_name": feature_names,
        "importance": importance,
        "rank": [int(np.where(sorted_idx == i)[0][0]) + 1 for i in range(obs_dim)],
    }).sort_values("importance", ascending=False)

    out_csv = PATHS.checkpoints / f"feature_importance_{MODEL_TYPE.lower()}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to: {out_csv}")

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()