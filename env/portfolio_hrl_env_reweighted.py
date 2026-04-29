"""
env/portfolio_hrl_env_reweighted.py

Synthetic pool sampler that draws paths with weights aligned to a target
modal-regime distribution rather than the pool's natural distribution.

Why: real test data is 38% SevereBear / 30% Bull / 30% Bear / 1% Crisis per
day, but synth pool's modal-regime distribution is 48% Bull / 36% Bear /
11% SB / 5% Crisis. The agent trains on Bull-modal paths but is evaluated
on a SB-heavy test period. Reweighting lets the agent see SB and Crisis
paths more often during training, matching the deployment distribution
(plus a Crisis boost since 1% test exposure is too thin to be informative).

Target distribution (modal-regime weighted):
  Bull-modal    : 25%
  Bear-modal    : 25%
  SevereBear    : 35%
  Crisis        : 15%

Implementation: classify each pool path by modal regime once at construction,
then on reset() draw a regime category from the target distribution and
sample a path uniformly within that category.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    PortfolioCore,
    SyntheticPoolCoreSampler,
)


# Target modal-regime distribution. Aligned to real-test per-day distribution
# (Bull 31%, Bear 30%, SB 38%, Crisis 1%) but with Crisis boosted to 15% since
# 1% in test is essentially zero exposure for evaluation purposes.
DEFAULT_TARGET_DISTRIBUTION = {
    0: 0.25,  # Bull
    1: 0.25,  # Bear
    2: 0.35,  # SevereBear
    3: 0.15,  # Crisis
}


class ReweightedSyntheticPoolCoreSampler(SyntheticPoolCoreSampler):
    """SyntheticPoolCoreSampler with non-uniform path sampling weighted by
    modal regime.

    Paths are classified by modal regime once at construction. On reset(),
    a modal regime is drawn from `target_distribution`, then a path is drawn
    uniformly from paths matching that modal regime.

    If a target regime has zero paths in the pool (shouldn't happen but worth
    guarding), that regime's weight is redistributed proportionally to the
    others.
    """

    def __init__(
        self,
        pool: dict,
        cfg: Optional[CoreConfig] = None,
        rng: Optional[np.random.Generator] = None,
        target_distribution: Optional[dict] = None,
    ):
        if pool.get("regimes") is None:
            raise ValueError(
                "Pool dict must include 'regimes' for the reweighted sampler. "
                "The synth pool .npz must have been built with regime labels."
            )

        target_distribution = target_distribution or DEFAULT_TARGET_DISTRIBUTION

        # Classify paths by modal regime BEFORE super().__init__() because
        # parent's __init__ calls self.reset(), which dispatches to our
        # reset() which uses self._path_indices_by_regime.
        pool_regimes = np.asarray(pool["regimes"]).astype(np.int64)
        n_paths = pool_regimes.shape[0]

        path_modal = np.empty(n_paths, dtype=np.int64)
        for p in range(n_paths):
            counts = np.bincount(pool_regimes[p], minlength=4)
            path_modal[p] = int(np.argmax(counts))

        # Group path indices by modal regime
        self._path_indices_by_regime = {
            r: np.where(path_modal == r)[0]
            for r in range(4)
        }

        # Build effective regime distribution after handling empty groups
        effective_dist = {}
        total_weight_active = 0.0
        for r, w in target_distribution.items():
            if len(self._path_indices_by_regime[r]) > 0:
                effective_dist[r] = w
                total_weight_active += w

        if total_weight_active == 0.0:
            raise ValueError(
                "All regime groups in target_distribution are empty in the pool"
            )

        # Renormalize over active regimes only
        self._regime_choices = []
        self._regime_weights = []
        for r, w in effective_dist.items():
            self._regime_choices.append(r)
            self._regime_weights.append(w / total_weight_active)
        self._regime_choices = np.array(self._regime_choices, dtype=np.int64)
        self._regime_weights = np.array(self._regime_weights, dtype=np.float64)

        # Diagnostics for the train script to log
        self.pool_modal_distribution = {
            r: float((path_modal == r).sum() / n_paths) for r in range(4)
        }
        self.target_distribution = dict(target_distribution)
        self.effective_distribution = {
            int(r): float(w) for r, w in zip(self._regime_choices, self._regime_weights)
        }

        # NOW call super().__init__() which will trigger reset()
        super().__init__(pool=pool, cfg=cfg, rng=rng)

    def reset(self, seed: Optional[int] = None) -> None:
        """Override to use weighted regime-then-path sampling.

        We delegate the per-path setup (PortfolioCore construction etc.) to
        the parent class via temporarily setting self.current_path before
        calling super().reset(). But super().reset() uses self.rng to pick
        the path. We bypass that by computing the path ourselves and writing
        it directly.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. Pick a modal regime from target distribution
        regime_idx = int(self.rng.choice(self._regime_choices, p=self._regime_weights))

        # 2. Pick a path uniformly within that regime's group
        candidates = self._path_indices_by_regime[regime_idx]
        chosen_path = int(self.rng.choice(candidates))

        # 3. Set up the core for this path. This mirrors the parent's reset()
        #    logic but with our chosen path instead of a uniform-random pick.
        self.current_path = chosen_path
        feats = self.pool_features[self.current_path]
        rets = self.pool_returns[self.current_path]
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        self._core = PortfolioCore(feats, rets, cfg=self.cfg, rng=self.rng)