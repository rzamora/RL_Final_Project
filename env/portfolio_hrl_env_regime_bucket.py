"""
env/portfolio_hrl_env_regime_bucket.py

LL training environment with regime-bucketed HL action sampling.

Disjoint rectangular partition of the [-1, +1]^2 HL action space, one
rectangle per regime. At episode start, we determine the modal regime
across the episode's path, and sample the HL action uniformly from
the corresponding rectangle. The HL action is held constant for the
episode (same as LowLevelPortfolioEnv and LowLevelPortfolioEnvRandomHL).

Bucket assignment (Option A — disjoint corners, deterministic):
  Crisis      -> gross_signal in [-1.0, -0.5], net_signal in [-1.0, -0.5]
  SevereBear  -> gross_signal in [-0.5,  0.0], net_signal in [-0.5,  0.0]
  Bear        -> gross_signal in [ 0.0, +0.5], net_signal in [ 0.0, +0.5]
  Bull        -> gross_signal in [+0.5, +1.0], net_signal in [+0.5, +1.0]

Together the four buckets cover the full diagonal of [-1, +1]^2 with no
overlap. Wrong combinations (e.g. Bull regime with gross=-1) never appear
in LL training data — exactly the design choice that distinguishes this
from LowLevelPortfolioEnvRandomHL.

This env requires that we know the regime label for every timestep in the
episode. Two construction patterns:
  (A) Wrap a real-data PortfolioCore. Pass `episode_regime_labels` as a
      1-D array aligned with the core's full data (one regime label per
      day), and the env will slice based on core.t_start and core.t_end
      after each reset.
  (B) Wrap a SyntheticRegimeBucketCoreSampler (defined below) which
      maintains the current path's regime sequence and exposes it via a
      `current_regimes` attribute after each reset.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    LowLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
)


# ---------------------------------------------------------------------------
# Bucket geometry (Option A)
# ---------------------------------------------------------------------------

# Each bucket: (gross_low, gross_high, net_low, net_high)
# regime_idx == 0 -> Bull, 1 -> Bear, 2 -> SevereBear, 3 -> Crisis
REGIME_BUCKETS = {
    0: (+0.5, +1.0, +0.5, +1.0),  # Bull
    1: (+0.0, +0.5, +0.0, +0.5),  # Bear
    2: (-0.5, +0.0, -0.5, +0.0),  # SevereBear
    3: (-1.0, -0.5, -1.0, -0.5),  # Crisis
}

REGIME_NAMES = ["Bull", "Bear", "SevereBear", "Crisis"]


def sample_from_bucket(regime_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Sample HL action [gross_signal, net_signal] uniformly from the
    rectangle corresponding to `regime_idx`."""
    g_lo, g_hi, n_lo, n_hi = REGIME_BUCKETS[regime_idx]
    gross = float(rng.uniform(g_lo, g_hi))
    net = float(rng.uniform(n_lo, n_hi))
    return np.array([gross, net], dtype=np.float32)


# ---------------------------------------------------------------------------
# Synth pool sampler that exposes regime sequences
# ---------------------------------------------------------------------------

class SyntheticRegimeBucketCoreSampler(SyntheticPoolCoreSampler):
    """SyntheticPoolCoreSampler that also tracks per-path regime labels.

    After reset(), `current_regimes` is a 1-D int array of regime labels
    for the active path (length = path_T). The env reads this to compute
    modal regime over the episode window.
    """

    def __init__(
        self,
        pool: dict,
        cfg: Optional[CoreConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if pool.get("regimes") is None:
            raise ValueError(
                "Pool dict must include 'regimes' (shape (n_paths, path_T)) "
                "for regime-bucket env."
            )
        # Must set this BEFORE super().__init__() because parent's __init__
        # calls self.reset() which dispatches to our reset() and needs
        # self.pool_regimes already in place.
        self.pool_regimes = np.asarray(pool["regimes"]).astype(np.int64)
        self.current_regimes = None  # set in reset()
        super().__init__(pool=pool, cfg=cfg, rng=rng)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        # super().reset() picked self.current_path
        self.current_regimes = self.pool_regimes[self.current_path]

# ---------------------------------------------------------------------------
# Regime-bucket LL env
# ---------------------------------------------------------------------------

class LowLevelPortfolioEnvRegimeBucketHL(LowLevelPortfolioEnv):
    """LL env with regime-bucketed HL action sampling per episode.

    On each reset():
      1. Core picks t_start, t_end (same as parent class)
      2. Read regime labels for the [t_start, t_end) window
      3. Compute modal regime
      4. Sample HL action from the modal regime's bucket
      5. Hold that HL action constant for the episode (parent class behavior)

    Construction:
      - core: a PortfolioCore (real data) OR a SyntheticRegimeBucketCoreSampler
      - episode_regime_labels: 1-D array (only required for real-data PortfolioCore;
        unused if core is a SyntheticRegimeBucketCoreSampler — that case uses
        core.current_regimes).
      - hl_rng: np.random.Generator for sampling within bucket.
    """

    def __init__(
        self,
        core,
        episode_regime_labels: Optional[np.ndarray] = None,
        hl_rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(core, fixed_hl_action=np.array([0.0, 0.0], dtype=np.float32))
        self.hl_rng = hl_rng or np.random.default_rng()

        # Determine source of regime labels
        if isinstance(core, SyntheticRegimeBucketCoreSampler):
            self._regime_source = "synth"
            self._episode_regime_labels = None
        elif episode_regime_labels is not None:
            self._regime_source = "real"
            self._episode_regime_labels = np.asarray(episode_regime_labels, dtype=np.int64)
            # Sanity: regime label length should match core's full data length
            if hasattr(core, "full_n_steps"):
                if len(self._episode_regime_labels) != core.full_n_steps:
                    raise ValueError(
                        f"episode_regime_labels length {len(self._episode_regime_labels)} "
                        f"does not match core.full_n_steps {core.full_n_steps}"
                    )
        else:
            raise ValueError(
                "LowLevelPortfolioEnvRegimeBucketHL requires either a "
                "SyntheticRegimeBucketCoreSampler core OR episode_regime_labels "
                "passed explicitly."
            )

        # Diagnostic: track which bucket was chosen on the last reset
        self.last_modal_regime: Optional[int] = None
        self.last_hl_action: Optional[np.ndarray] = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            # Offset so HL RNG and core RNG don't share state
            self.hl_rng = np.random.default_rng(seed + 7919)

        # Reset core first — this picks t_start / t_end
        obs_packet = super().reset(seed=seed, options=options)
        # super().reset() returns (obs, info); we'll re-emit obs after we set HL

        # Read regime labels for the episode window
        if self._regime_source == "synth":
            window_regimes = self.core.current_regimes[
                self.core.t_start : self.core.t_end
            ]
        else:
            window_regimes = self._episode_regime_labels[
                self.core.t_start : self.core.t_end
            ]

        # Compute modal regime
        # Safety: if window is empty (shouldn't happen), fall back to a uniform
        # sample over all buckets (treats it as "unknown regime")
        if len(window_regimes) == 0:
            modal = int(self.hl_rng.integers(0, 4))
        else:
            counts = np.bincount(window_regimes, minlength=4)
            modal = int(np.argmax(counts))

        self.last_modal_regime = modal
        new_hl = sample_from_bucket(modal, self.hl_rng)
        self.last_hl_action = new_hl
        self.fixed_hl_action = new_hl

        # Re-emit the observation with the updated HL action appended
        # (super().reset() already constructed obs with the placeholder HL)
        return self._get_obs(), obs_packet[1]