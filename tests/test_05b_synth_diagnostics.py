# tests/test_05b_synth_diagnostics.py
"""
Synth pool diagnostics — the two checks the smoke test doesn't cover:

  1. No lookahead leakage on synth data. Oracle that peeks at returns[t+1]
     should massively beat 1.0; anti-oracle should crater.

  2. Regime distribution. Confirms the pool isn't degenerate and prints the
     per-class share so you can compare against your known encoding.
     (Hard threshold can be added once we know which int = SevereBear.)

Run:
    pytest tests/test_05b_synth_diagnostics.py -v -s
"""

from collections import Counter

import numpy as np
import pytest

from project_config import PATHS
from env.portfolio_hrl_env_fixed import (
    CoreConfig,
    LowLevelPortfolioEnv,
    PortfolioCore,
    SyntheticPoolCoreSampler,
    load_synthetic_pool,
)


# Path length is 384; PortfolioCore requires full_n_steps >= episode_length + 1,
# so we run episodes of 383 steps. Matches what test_05 already uses.
SYNTH_EPISODE_LENGTH = 383


@pytest.fixture(scope="module")
def pool():
    return load_synthetic_pool(PATHS.synth_pool)


@pytest.fixture(scope="module")
def cfg():
    return CoreConfig(episode_length=SYNTH_EPISODE_LENGTH)


# ---------------------------------------------------------------------------
# 1. Lookahead diagnostic — oracle vs anti-oracle
# ---------------------------------------------------------------------------

def _run_episode(env, policy_fn):
    """Run one full episode with action = policy_fn(env.core). Return final equity."""
    env.reset(seed=7)
    final_equity = None
    while True:
        action = policy_fn(env.core)
        _, _, terminated, truncated, info = env.step(action)
        final_equity = info["equity"]
        if terminated or truncated:
            break
    return final_equity


def test_no_lookahead_on_synth(pool, cfg):
    """Oracle peeks at returns[t+1] and bets on it; anti-oracle bets against it.
    On the same return sequence, the two should diverge dramatically. If they
    don't, apply_allocation has an off-by-one bug.
    """
    fixed_path_idx = 0
    feats = np.nan_to_num(pool["features"][fixed_path_idx])
    rets = pool["returns"][fixed_path_idx]

    print(f"\n[path {fixed_path_idx} returns scale] "
          f"max|r|={np.abs(rets).max():.4f}  std={rets.std():.4f}")

    def make_env():
        core = PortfolioCore(feats, rets, cfg=cfg, rng=np.random.default_rng(0))
        return LowLevelPortfolioEnv(core)

    def oracle(core):
        next_t = core.t + 1
        if next_t >= len(core.returns):
            return np.zeros(core.n_assets, dtype=np.float32)
        r = core.returns[next_t]
        best = int(np.argmax(np.abs(r)))
        a = np.zeros(core.n_assets, dtype=np.float32)
        a[best] = np.sign(r[best]) * 1.0
        return a

    def anti_oracle(core):
        next_t = core.t + 1
        if next_t >= len(core.returns):
            return np.zeros(core.n_assets, dtype=np.float32)
        r = core.returns[next_t]
        worst = int(np.argmax(np.abs(r)))
        a = np.zeros(core.n_assets, dtype=np.float32)
        a[worst] = -np.sign(r[worst]) * 1.0
        return a

    oracle_eq = _run_episode(make_env(), oracle)
    anti_eq = _run_episode(make_env(), anti_oracle)
    ratio = oracle_eq / max(anti_eq, 1e-9)

    print(
        f"[lookahead diagnostic on synth path {fixed_path_idx}] "
        f"oracle={oracle_eq:.3g}  anti={anti_eq:.3g}  ratio={ratio:.1f}x"
    )

    assert oracle_eq > 1.5, (
        f"Oracle final equity {oracle_eq:.3f} too low. Either env is NOT "
        f"using next-step returns (defeats the t+1 fix) or path scale is off."
    )
    assert anti_eq < 0.7, (
        f"Anti-oracle final equity {anti_eq:.3f} too high. Anti-oracle should "
        f"crash — if it doesn't, weights may be applied to returns[t] (already "
        f"known) instead of returns[t+1]."
    )
    assert ratio > 5.0, (
        f"Oracle/anti ratio {ratio:.2f}x too small. Perfect-foresight and "
        f"perfect-hindsight should diverge dramatically on the same returns."
    )


# ---------------------------------------------------------------------------
# 2. Regime distribution
# ---------------------------------------------------------------------------

def test_regime_distribution(pool, cfg):
    """Sample paths through the sampler (so we test what training will see)
    and report the distribution of modal per-path regime labels.

    Labels in this pool are int8 per-step (shape n_paths × T). We summarize
    each path by its modal label and print the resulting histogram.

    Hard check: distribution is not degenerate (more than one class).
    Operator check: eyeball the histogram against your known encoding to
    confirm SevereBear/Crisis classes are over-represented vs the real-CSV
    train share of 16.4%.
    """
    if pool["regimes"] is None:
        pytest.skip("Pool has no 'regimes' array")

    regimes = pool["regimes"]
    print(f"\n[regimes array] shape={regimes.shape}  dtype={regimes.dtype}  "
          f"unique values={sorted(np.unique(regimes).tolist())}")

    sampler = SyntheticPoolCoreSampler(pool, cfg=cfg, rng=np.random.default_rng(0))

    n_samples = 500
    modal_labels = []
    per_step_counter = Counter()  # also track per-step share, not just modal
    for _ in range(n_samples):
        sampler.reset()
        path_regimes = np.asarray(regimes[sampler.current_path])
        vals, counts = np.unique(path_regimes, return_counts=True)
        modal = int(vals[np.argmax(counts)])
        modal_labels.append(modal)
        for v, c in zip(vals, counts):
            per_step_counter[int(v)] += int(c)

    modal_counter = Counter(modal_labels)
    total_paths = sum(modal_counter.values())
    total_steps = sum(per_step_counter.values())

    print(f"\n[modal regime over {n_samples} sampled paths]")
    for k in sorted(modal_counter):
        v = modal_counter[k]
        print(f"  class {k}: {v} paths ({v/total_paths:.1%})")

    print(f"\n[per-step regime share over {n_samples} sampled paths × "
          f"{cfg.episode_length+1} steps each]")
    for k in sorted(per_step_counter):
        v = per_step_counter[k]
        print(f"  class {k}: {v} steps ({v/total_steps:.1%})")

    print(
        "\n[interpretation] Compare per-step shares above against train CSV "
        "(SevereBear was 16.4%) and test CSV (37.8%). The synth pool was "
        "designed to over-sample SevereBear, so whichever class is "
        "SevereBear should be well above 16.4% per-step here."
    )

    assert len(modal_counter) >= 2, (
        f"Only one modal regime across {n_samples} paths — pool degenerate "
        f"or sampler biased. Counter: {modal_counter}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))