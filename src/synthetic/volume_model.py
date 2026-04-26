"""
Volume modeling for the regime-switching DCC-GARCH-copula synthetic generator.

Adds a regime-conditional log-volume model to the existing pipeline:

    log(V_t) = α_r + β_pos_r · max(r_t, 0)
              + β_neg_r · min(r_t, 0)
              + γ_r · log(V_{t-1})
              + σ_r · ε_t

where ε_t ~ N(0, 1) and the parameters depend on regime r and asset.

The asymmetric coefficients (β_pos vs β_neg) capture the well-documented
asymmetry where down-days often see stronger volume responses than up-days
(MDH and "no news on no-volume rallies" effects).

Per-regime fitting captures the fact that volume-return coupling differs
across regimes — Crisis days have stronger |return|→volume responses than
Bull days.

Usage
-----
1. Load volumes alongside returns in download_data()
2. Call fit_volume_models_per_regime() inside fit_per_regime()
3. Call simulate_volume_step() inside the simulate_hybrid_paths() loop
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading volumes (extension of download_data)
# ---------------------------------------------------------------------------
def load_volumes_from_csvs(data_dir, assets, dates_index, file_pattern="{ticker}_StockData_RL.csv"):
    """
    Load Volume column for each asset, aligned to the given dates_index.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the per-asset CSVs.
    assets : list[str]
        Tickers to load.
    dates_index : pd.DatetimeIndex
        The aligned date index from download_data() (use returns.index).
    file_pattern : str
        Filename pattern with '{ticker}' placeholder.

    Returns
    -------
    volumes : pd.DataFrame
        Columns = assets, index = dates_index, values = raw daily volume.
    """
    import os

    frames = []
    for ticker in assets:
        path = os.path.join(data_dir, file_pattern.format(ticker=ticker))
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"])

        if "Volume" not in df.columns:
            raise ValueError(f"{ticker} CSV missing 'Volume' column")

        v = (
            df[["DATE", "Volume"]]
            .rename(columns={"Volume": ticker})
            .set_index("DATE")
            .sort_index()
        )
        frames.append(v)

    volumes = pd.concat(frames, axis=1, join="inner")
    volumes = volumes.reindex(dates_index)

    # Defensive cleanup
    volumes = volumes.replace(0, np.nan).ffill().bfill()
    if volumes.isna().any().any():
        raise ValueError("Volume DataFrame still contains NaN after cleanup")

    return volumes


# ---------------------------------------------------------------------------
# Volume model fitting
# ---------------------------------------------------------------------------
def fit_volume_model_single(returns, volume, asset_name="?"):
    """
    Fit asymmetric AR(1) log-volume model for one asset, on the returns/volume
    subset belonging to a single regime.

    Model:
        log(V_t) = α + β_pos·max(r_t, 0) + β_neg·min(r_t, 0) + γ·log(V_{t-1}) + σ·ε_t

    Parameters
    ----------
    returns : pd.Series
        Daily returns in % (matches the rest of the pipeline).
    volume : pd.Series
        Daily volume (raw shares/contracts).
    asset_name : str
        For diagnostic messaging only.

    Returns
    -------
    model : dict
        Keys: alpha, beta_pos, beta_neg, gamma, sigma, mean_log_vol.
        mean_log_vol is used as a fallback initialization in simulation.
    """
    log_vol = np.log(np.maximum(volume.values, 1.0))
    ret = returns.values

    if len(ret) != len(log_vol):
        raise ValueError(f"{asset_name}: returns and volume length mismatch")

    # Construct lagged log volume
    log_vol_lag = np.roll(log_vol, 1)
    log_vol_lag[0] = log_vol_lag[1]   # benign warm-up

    pos_part = np.clip(ret, 0, None)
    neg_part = np.clip(ret, None, 0)

    # OLS via normal equations: y = X @ b
    X = np.column_stack([
        np.ones(len(ret)),
        pos_part,
        neg_part,
        log_vol_lag,
    ])
    y = log_vol

    # Solve with regularization to avoid issues when pos/neg parts are sparse
    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-6
    Xty = X.T @ y
    coeffs = np.linalg.solve(XtX, Xty)
    alpha, beta_pos, beta_neg, gamma = coeffs

    # Residual sigma
    resid = y - X @ coeffs
    sigma = float(np.std(resid, ddof=X.shape[1]))

    # Stability guard: if AR(1) coefficient is too close to 1, the simulated
    # log-volume can drift unboundedly. Cap it.
    gamma = float(np.clip(gamma, 0.0, 0.99))

    return {
        "alpha": float(alpha),
        "beta_pos": float(beta_pos),
        "beta_neg": float(beta_neg),
        "gamma": gamma,
        "sigma": float(max(sigma, 1e-4)),
        "mean_log_vol": float(np.mean(log_vol)),
        "last_log_vol": float(log_vol[-1]),
    }


def fit_volume_models_per_regime(returns, volumes, regime_seq, n_regimes, assets, min_obs=60):
    """
    Fit per-asset volume models within each regime.

    Returns
    -------
    volume_models : list[dict[str, dict]]
        volume_models[regime_idx][asset] = {alpha, beta_pos, beta_neg, gamma, sigma, ...}
    """
    print("\nFitting per-regime volume models...")
    volume_models = []

    for k in range(n_regimes):
        mask = regime_seq == k
        obs = int(mask.sum())

        if obs < min_obs:
            print(f"  Regime {k}: too few obs ({obs}), placeholder")
            volume_models.append(None)
            continue

        ret_k = returns.loc[mask]
        vol_k = volumes.loc[mask]

        regime_vm = {}
        for asset in assets:
            vm = fit_volume_model_single(ret_k[asset], vol_k[asset], asset_name=asset)
            regime_vm[asset] = vm
            print(
                f"  Regime {k} ({asset}): α={vm['alpha']:.3f} "
                f"β+={vm['beta_pos']:.4f} β-={vm['beta_neg']:.4f} "
                f"γ={vm['gamma']:.3f} σ={vm['sigma']:.3f}"
            )

        volume_models.append(regime_vm)

    # Fill sparse regimes
    fill_model = next((m for m in volume_models if m is not None), None)
    if fill_model is None:
        raise RuntimeError("No regime had enough observations for volume model fitting")

    for i in range(len(volume_models)):
        if volume_models[i] is None:
            volume_models[i] = fill_model
            print(f"  Regime {i}: filled with another regime's volume model")

    return volume_models


# ---------------------------------------------------------------------------
# Simulation step
# ---------------------------------------------------------------------------
def simulate_volume_step(ret_t, log_vol_prev, volume_model, rng=None):
    """
    Simulate one step of log-volume given the realized return.

    Parameters
    ----------
    ret_t : float
        Today's return for this asset (in %).
    log_vol_prev : float
        Yesterday's log-volume (state variable).
    volume_model : dict
        Fitted parameters for this asset and regime.
    rng : np.random.Generator or None
        RNG for reproducibility.

    Returns
    -------
    log_vol_new : float
        Today's log-volume.
    volume_new : float
        Today's volume in raw units.
    """
    if rng is None:
        eps = np.random.normal(0, 1)
    else:
        eps = rng.standard_normal()

    pos = max(ret_t, 0.0)
    neg = min(ret_t, 0.0)

    log_vol_new = (
        volume_model["alpha"]
        + volume_model["beta_pos"] * pos
        + volume_model["beta_neg"] * neg
        + volume_model["gamma"] * log_vol_prev
        + volume_model["sigma"] * eps
    )

    # Clamp to a reasonable range to prevent pathological tails
    # Allow ±5 sigma above/below the typical level
    lower = volume_model["mean_log_vol"] - 5 * volume_model["sigma"] / max(1 - volume_model["gamma"], 0.05)
    upper = volume_model["mean_log_vol"] + 5 * volume_model["sigma"] / max(1 - volume_model["gamma"], 0.05)
    log_vol_new = float(np.clip(log_vol_new, lower, upper))

    return log_vol_new, float(np.exp(log_vol_new))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def diagnose_volume_relationships(returns, volumes, regime_seq, assets):
    """
    Empirical check of volume-return coupling per regime.
    Prints a table for each asset showing |ret|–log(vol) correlation and
    log(vol) AR(1) per regime. Useful to validate before fitting.
    """
    print("\nVolume diagnostic — empirical relationships per regime per asset:")
    log_vol = np.log(np.maximum(volumes, 1.0))
    abs_ret = returns.abs()

    for asset in assets:
        print(f"\n  {asset}:")
        print(f"    {'Regime':<8s} {'|ret|-log(vol) corr':>22s} {'log(vol) AR(1)':>18s} {'asymm':>10s}")
        for r in range(int(regime_seq.max()) + 1):
            mask = regime_seq == r
            if mask.sum() < 30:
                continue
            corr_abs = abs_ret[asset][mask].corr(log_vol[asset][mask])
            ar1 = log_vol[asset][mask].autocorr(1)

            # Asymmetry: difference in mean log(vol) between up and down days
            up_mask = mask & (returns[asset] > 0)
            dn_mask = mask & (returns[asset] < 0)
            asymm = log_vol[asset][dn_mask].mean() - log_vol[asset][up_mask].mean()

            print(f"    {r:<8d} {corr_abs:22.3f} {ar1:18.3f} {asymm:10.3f}")


def diagnose_simulated_volume(sim_returns, sim_volumes, hist_returns, hist_volumes, assets):
    """
    Compare simulated vs historical volume statistics.
    Call this on a single representative path after simulation.

    Parameters
    ----------
    sim_returns, sim_volumes : np.ndarray, shape (n_steps, N)
        One simulated path.
    hist_returns, hist_volumes : pd.DataFrame
        Historical data for comparison.
    """
    print("\nSimulated vs historical volume diagnostics:")
    print(f"  {'Asset':<6s} {'Hist mean log(V)':>18s} {'Sim mean log(V)':>18s} "
          f"{'Hist corr':>12s} {'Sim corr':>12s}")

    for i, asset in enumerate(assets):
        h_logv = np.log(np.maximum(hist_volumes[asset].values, 1.0))
        s_logv = np.log(np.maximum(sim_volumes[:, i], 1.0))

        h_absret = np.abs(hist_returns[asset].values)
        s_absret = np.abs(sim_returns[:, i])

        h_corr = np.corrcoef(h_absret, h_logv)[0, 1]
        s_corr = np.corrcoef(s_absret, s_logv)[0, 1]

        print(f"  {asset:<6s} {h_logv.mean():18.3f} {s_logv.mean():18.3f} "
              f"{h_corr:12.3f} {s_corr:12.3f}")
