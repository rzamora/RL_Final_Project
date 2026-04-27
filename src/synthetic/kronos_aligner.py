"""
Real-to-synthetic Kronos feature alignment.

Strategy
--------
The synthetic generator produces returns and regime labels per path, but it
cannot produce Kronos forecast features (Kronos is a foundation model trained
on real OHLCV — it doesn't apply to synthetic data).

This module aligns real Kronos features to synthetic paths by matching real
historical date windows to synthetic regime sequences. Each synthetic path
gets a copy of real Kronos features from a real period that matches the
synthetic regime trajectory.

Two strategies:
    'random'       — uniformly sample a random L-day window from the real data
    'regime_match' — score real windows by regime overlap with the synthetic path,
                     pick the best match (sampled probabilistically among top-K)

Usage
-----
    aligner = KronosAligner(real_features_df, real_regimes, kronos_columns)
    
    for path_idx in range(n_paths):
        synth_regimes = all_regimes[path_idx]   # shape (252,)
        kronos_block = aligner.assign(synth_regimes, strategy='regime_match')
        # kronos_block: DataFrame of shape (252, n_kronos_cols)
        # — append it as columns to your synthetic features DataFrame
"""

import numpy as np
import pandas as pd


class KronosAligner:
    """
    Aligns real Kronos features to synthetic paths.

    Parameters
    ----------
    real_features_df : pd.DataFrame
        Real-data merged DataFrame with at least:
        - the Kronos columns to be transferred
        - a regime column matching the synthetic regime labels
    real_regime_seq : np.ndarray of int, shape (n_real_dates,)
        Regime labels at each real date. Same regime encoding as the
        synthetic generator (e.g., 0=Bull, 1=Bear, 2=SevereBear, 3=Crisis).
    kronos_columns : list[str]
        Column names in real_features_df representing Kronos features.
        These are what gets copied into each synthetic path.
    seed : int, optional
        For reproducibility.
    """

    def __init__(self, real_features_df, real_regime_seq, kronos_columns, seed=None):
        self.df = real_features_df.reset_index(drop=True).copy()
        self.regimes = np.asarray(real_regime_seq, dtype=int)
        self.kronos_columns = list(kronos_columns)
        self.rng = np.random.default_rng(seed)

        # Validate
        if len(self.df) != len(self.regimes):
            raise ValueError(
                f"real_features_df has {len(self.df)} rows but "
                f"real_regime_seq has {len(self.regimes)} entries."
            )
        missing = [c for c in self.kronos_columns if c not in self.df.columns]
        if missing:
            raise KeyError(f"Missing Kronos columns in real_features_df: {missing}")

    # -------------------------------------------------------------------------
    # Strategy: random window
    # -------------------------------------------------------------------------
    def _random_window(self, n_steps):
        """Pick a random L-day window from real data."""
        max_start = len(self.df) - n_steps
        if max_start < 0:
            raise ValueError(
                f"n_steps ({n_steps}) exceeds real data length ({len(self.df)})"
            )
        start = int(self.rng.integers(0, max_start + 1))
        return start

    # -------------------------------------------------------------------------
    # Strategy: regime-matched window
    # -------------------------------------------------------------------------
    def _regime_match_window(self, synth_regime_seq, top_k=5, sample_temperature=1.0):
        """
        Find real windows whose regime sequences best match the synthetic path.
        Returns the start index of one of the top-K matches, sampled with weights.

        Parameters
        ----------
        synth_regime_seq : np.ndarray, shape (n_steps,)
        top_k : int
            Number of best-matching windows to consider.
        sample_temperature : float
            Higher = more uniform sampling among top-K, lower = greedy.
            1.0 = use raw match scores as weights.
        """
        n_steps = len(synth_regime_seq)
        max_start = len(self.df) - n_steps
        if max_start < 0:
            raise ValueError(
                f"n_steps ({n_steps}) exceeds real data length ({len(self.df)})"
            )

        # Compute match score for each candidate start position
        # Vectorized: count how many timesteps share regime
        n_starts = max_start + 1

        # Use a sliding-window comparison
        match_counts = np.zeros(n_starts, dtype=int)
        for i in range(n_steps):
            # At synthetic step i, real regime at start+i should match synth_regime_seq[i]
            real_at_offset_i = self.regimes[i : i + n_starts]
            match_counts += (real_at_offset_i == synth_regime_seq[i]).astype(int)

        # Pick top-K matching windows
        top_k_actual = min(top_k, n_starts)
        top_indices = np.argpartition(-match_counts, top_k_actual - 1)[:top_k_actual]

        # Weighted sampling among top-K (favor higher matches)
        weights = match_counts[top_indices].astype(float)
        if sample_temperature > 0:
            weights = weights ** (1.0 / sample_temperature)
        weights = weights / (weights.sum() + 1e-12)

        chosen = self.rng.choice(top_indices, p=weights)
        return int(chosen), int(match_counts[chosen]), n_steps

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def assign(self, synth_regime_seq, strategy='regime_match', **kwargs):
        """
        Get a Kronos feature block for one synthetic path.

        Parameters
        ----------
        synth_regime_seq : np.ndarray, shape (n_steps,)
            Regime labels for the synthetic path.
        strategy : {'random', 'regime_match'}
        **kwargs : passed to the strategy function

        Returns
        -------
        kronos_block : pd.DataFrame, shape (n_steps, n_kronos_cols)
            Indexed 0..n_steps-1. Same column order as `kronos_columns`.
        """
        n_steps = len(synth_regime_seq)

        if strategy == 'random':
            start = self._random_window(n_steps)
            match_score = None
        elif strategy == 'regime_match':
            start, match_score, _ = self._regime_match_window(
                synth_regime_seq, **kwargs
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        block = self.df[self.kronos_columns].iloc[start : start + n_steps].copy()
        block = block.reset_index(drop=True)

        # Attach metadata for debugging
        block.attrs['source_start_idx'] = start
        block.attrs['match_score'] = match_score
        block.attrs['match_pct'] = (
            match_score / n_steps if match_score is not None else None
        )

        return block

    def diagnose_match_quality(self, synth_regime_seq, n_trials=10):
        """
        Sample multiple regime-matched windows and report match quality.
        Useful to validate that the alignment is working well.
        """
        scores = []
        for _ in range(n_trials):
            block = self.assign(synth_regime_seq, strategy='regime_match')
            scores.append(block.attrs['match_pct'])
        scores = np.array(scores)
        return {
            'mean_match_pct': float(scores.mean()),
            'min_match_pct': float(scores.min()),
            'max_match_pct': float(scores.max()),
            'n_steps': len(synth_regime_seq),
        }


# -------------------------------------------------------------------------
# Demo
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Pretend we have real data
    np.random.seed(42)
    n_real = 4000
    real_df = pd.DataFrame({
        'date': pd.bdate_range('2005-01-01', periods=n_real),
        'kronos_close_d5': np.random.randn(n_real) * 0.02,
        'kronos_conf': np.random.rand(n_real),
        'kronos_error_stab': np.random.rand(n_real),
    })
    real_regimes = np.random.choice([0, 1, 2, 3], size=n_real, p=[0.4, 0.4, 0.15, 0.05])

    aligner = KronosAligner(
        real_features_df=real_df,
        real_regime_seq=real_regimes,
        kronos_columns=['kronos_close_d5', 'kronos_conf', 'kronos_error_stab'],
        seed=0,
    )

    # Simulate a synthetic path with mostly Crisis regimes
    synth_path = np.array([3] * 100 + [1] * 100 + [0] * 52)

    # Random alignment
    rand_block = aligner.assign(synth_path, strategy='random')
    print(f"Random block shape: {rand_block.shape}")
    print(f"  Source start idx: {rand_block.attrs['source_start_idx']}")

    # Regime-matched alignment
    rm_block = aligner.assign(synth_path, strategy='regime_match', top_k=5)
    print(f"\nRegime-matched block shape: {rm_block.shape}")
    print(f"  Source start idx: {rm_block.attrs['source_start_idx']}")
    print(f"  Match score: {rm_block.attrs['match_score']}/{len(synth_path)} "
          f"({rm_block.attrs['match_pct']:.1%})")

    # Diagnostic
    diag = aligner.diagnose_match_quality(synth_path, n_trials=20)
    print(f"\nMatch quality across 20 trials:")
    for k, v in diag.items():
        print(f"  {k}: {v}")
