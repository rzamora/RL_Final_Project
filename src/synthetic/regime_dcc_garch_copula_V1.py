
"""
Version of regime_dcc_garch_copula.py
=============================================
- 4 regimes  Bull / Bear / Crisis / InflationShock
- Economically-aware regime relabeling using:
    equity means + TLT mean + volatility proxy
- Helper functions:
    classify_regime_labels
    regime_summary_table
    build_synthetic_prices
- fit_per_regime stores summary statistics
- simulate_hybrid_paths supports:
    * random starting regime
    * stress-biased transition probabilities
    * optional synthetic price reconstruction
- simple demo environment retained but renamed to SimpleRegimeAwareMarketEnv
  (not intended as the final project environment)

Important:
- Shorting should still be handled in the RL environment via negative weights.
- The synthetic generator only simulates market returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import pickle
import os
warnings.filterwarnings("ignore")

from arch import arch_model
from arch.univariate import StudentsT
from scipy.stats import t as student_t
from scipy.special import gammaln
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Model is not converging")

# optional HMM backend
try:
    from hmmlearn import hmm as hmmlearn_hmm
    from hmmlearn import _utils  # if needed
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    print("hmmlearn not found — using fallback clustering. Install with: pip install hmmlearn")

from volume_model import (
    load_volumes_from_csvs,
    fit_volume_model_single,         # ← add this
    fit_volume_models_per_regime,    # can keep or remove (no longer used)
    simulate_volume_step,
    diagnose_volume_relationships,
    diagnose_simulated_volume,
)
# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_SYN = PROJECT_ROOT / "data" / "synthetic"
DATA_SYN_MOD = PROJECT_ROOT / "data" / "synthetic" / "models"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ASSETS       = ["NVDA", "AMD", "SMH", "TLT"]
TBILL_TICKER = "DTB3"
START        = "2004-02-07"  # TLT has the least history, this is the start date of TLT
END          = "2022-12-31"  # set so we can test 2023, 2024, 2025 and 2026

# 4-regime design:
#   0 = Bull
#   1 = Bear
#   2 = Crisis
#   3 = InflationShock
N_REGIMES    = 4
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

REGIME_NAMES = {
    0: "Bull",
    1: "Bear",
    2: "Severe Bear",
    3: "Crisis",
}
REGIME_COLORS = {
    0: "#2ecc71",   # green
    1: "#f39c12",   # orange (mild stress)
    2: "#e67e22",   # darker orange (severe)
    3: "#e74c3c",   # red (acute crisis)
}


# =============================================================================
# 1. DATA
# =============================================================================
def download_data(assets, tbill_ticker, start=None, end=None):
    """
    Load already-downloaded RL market data from CSV files stored in the repo's
    relative data directory.

    Expected files in data_dir:
        NVDA_StockData_RL.csv
        AMD_StockData_RL.csv
        SMH_StockData_RL.csv
        TLT_StockData_RL.csv
        DTB3_StockData_RL.csv

    Risky asset file format:
        DATE, TICKER, open, close, high, low, Volume, VWAP, CPct_LChg

    DTB3 file format:
        DATE, TICKER, rate, CPct_Chg

    Parameters
    ----------
    assets : list[str]
        Example: ['NVDA', 'AMD', 'SMH', 'TLT']
    tbill_ticker : str
        Example: 'DTB3'
    start : str or None
        Optional start date filter, e.g. '2015-01-01'
    end : str or None
        Optional end date filter, e.g. '2024-12-31'
    data_dir : str
        Relative directory containing CSV files. Defaults to 'data/raw'.

    Returns
    -------
    prices : pd.DataFrame
        Close price matrix for risky assets
    returns : pd.DataFrame
        Daily log return matrix (%) for risky assets, from CPct_LChg
    tbill_d : pd.Series
        Daily short-rate return series, from DTB3 CPct_Chg
    """
    print("Reading files...")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    price_frames = []
    return_frames = []

    # -------------------------
    # Load risky assets
    # -------------------------
    for ticker in assets:
        file_path = os.path.join(DATA_DIR, f"{ticker}_StockData_RL.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE")

        required_cols = {"DATE", "close", "CPct_LChg"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{ticker} file missing columns: {missing}")

        if start is not None:
            df = df[df["DATE"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["DATE"] <= pd.to_datetime(end)]

        px = (
            df[["DATE", "close"]]
            .rename(columns={"close": ticker})
            .set_index("DATE")
        )

        rt = (
            df[["DATE", "CPct_LChg"]]
            .rename(columns={"CPct_LChg": ticker})
            .set_index("DATE")
        )

        price_frames.append(px)
        return_frames.append(rt)

    prices = pd.concat(price_frames, axis=1, join="inner").sort_index()
    returns = pd.concat(return_frames, axis=1, join="inner").sort_index()

    # -------------------------
    # Load DTB3 / short rate
    # -------------------------
    tbill_path = os.path.join(DATA_DIR, f"{tbill_ticker}_StockData_RL.csv")
    if not os.path.exists(tbill_path):
        raise FileNotFoundError(f"Missing file: {tbill_path}")

    tbill = pd.read_csv(tbill_path)
    tbill.columns = [c.strip() for c in tbill.columns]
    tbill["DATE"] = pd.to_datetime(tbill["DATE"])
    tbill = tbill.sort_values("DATE")

    required_tbill_cols = {"DATE", "CPct_Chg"}
    missing_tbill = required_tbill_cols - set(tbill.columns)
    if missing_tbill:
        raise ValueError(f"{tbill_ticker} file missing columns: {missing_tbill}")

    if start is not None:
        tbill = tbill[tbill["DATE"] >= pd.to_datetime(start)]
    if end is not None:
        tbill = tbill[tbill["DATE"] <= pd.to_datetime(end)]

    tbill_d = (
        tbill[["DATE", "CPct_Chg"]]
        .rename(columns={"CPct_Chg": tbill_ticker})
        .set_index("DATE")[tbill_ticker]
    )

    # -------------------------
    # Align all to common dates
    # -------------------------
    common = prices.index.intersection(returns.index).intersection(tbill_d.index)
    prices = prices.loc[common]
    returns = returns.loc[common]
    tbill_d = tbill_d.loc[common]

    if len(returns) == 0:
        raise ValueError("No overlapping dates found after alignment.")

    print(f"  {len(returns)} trading days: {returns.index[0].date()} → {returns.index[-1].date()}")
    return prices, returns, tbill_d


# =============================================================================
# 2. HELPERS
# =============================================================================
def classify_regime_labels(means, covars, assets):
    """
    Label regimes based on what the data actually shows.
    Priority: Bull (+drift) → Crisis (highest vol) → Bear (negative eq) → SevereBear (rest)
    """
    if assets is None:
        raise ValueError("assets must be provided")
    asset_idx = {a: i for i, a in enumerate(assets)}
    eq_idx = [asset_idx["NVDA"], asset_idx["AMD"], asset_idx["SMH"]]
    tlt_idx = asset_idx["TLT"]

    scores = []
    for k in range(means.shape[0]):
        scores.append({
            "raw": k,
            "eq_mean": float(means[k, eq_idx].mean()),
            "tlt_mean": float(means[k, tlt_idx]),
            "eq_vol": float(np.sqrt(np.diag(covars[k])[eq_idx].mean())),
        })

    # Bull = highest equity mean (regardless of vol — could be high-vol AI bull)
    bull = max(scores, key=lambda d: d["eq_mean"])["raw"]
    remaining = [s for s in scores if s["raw"] != bull]

    # Crisis = highest equity vol among non-Bull regimes
    crisis = max(remaining, key=lambda d: d["eq_vol"])["raw"]
    remaining = [s for s in remaining if s["raw"] != crisis]

    # SevereBear = next-highest vol with negative equity mean
    severe = max(remaining, key=lambda d: d["eq_vol"] - d["eq_mean"])["raw"]
    bear = [s for s in remaining if s["raw"] != severe][0]["raw"]

    return {bull: 0, bear: 1, severe: 2, crisis: 3}

def regime_summary_table(regime_seq, returns):
    """
    Summary table by canonical regime label.
    """
    rows = []
    for k in range(N_REGIMES):
        mask = regime_seq == k
        if mask.sum() == 0:
            rows.append({
                "regime": REGIME_NAMES[k],
                "n_obs": 0,
                "eq_mean": np.nan,
                "tlt_mean": np.nan,
                "avg_vol": np.nan,
            })
            continue

        sub = returns.loc[mask]
        eq_mean = sub[["NVDA", "AMD", "SMH"]].mean().mean()
        tlt_mean = sub["TLT"].mean()
        avg_vol = sub.std().mean()

        rows.append({
            "regime": REGIME_NAMES[k],
            "n_obs": int(mask.sum()),
            "eq_mean": float(eq_mean),
            "tlt_mean": float(tlt_mean),
            "avg_vol": float(avg_vol),
        })

    return pd.DataFrame(rows)


def make_regime_features(returns, window=21):
    """Rolling features that describe regime state."""
    eq = returns[["NVDA", "AMD", "SMH"]].mean(axis=1)
    tlt = returns["TLT"]

    feats = pd.DataFrame(index=returns.index)
    feats["eq_mean"]     = eq.rolling(window).mean()
    feats["eq_vol"]      = eq.rolling(window).std()
    feats["tlt_mean"]    = tlt.rolling(window).mean()
    feats["tlt_vol"]     = tlt.rolling(window).std()
    feats["eq_tlt_corr"] = eq.rolling(window).corr(tlt)
    feats["downside"]    = eq.clip(upper=0).rolling(window).mean()
    return feats

def build_synthetic_prices(sim_returns, start_price=100.0):
    """
    Convert simulated returns (%) into normalized price paths.

    Parameters
    ----------
    sim_returns : np.ndarray
        shape = (n_paths, n_steps, n_assets), returns in percent
    start_price : float
        initial price level

    Returns
    -------
    sim_prices : np.ndarray
        shape = (n_paths, n_steps + 1, n_assets)
    """
    ret = np.asarray(sim_returns, dtype=float) / 100.0
    n_paths, n_steps, n_assets = ret.shape
    prices = np.zeros((n_paths, n_steps + 1, n_assets), dtype=float)
    prices[:, 0, :] = start_price
    for t in range(n_steps):
        prices[:, t + 1, :] = prices[:, t, :] * (1.0 + ret[:, t, :])
    return prices

def standardized_t_cdf(z, nu):
    """CDF of standardized Student-t (var=1) at z."""
    scale = np.sqrt(nu / (nu - 2))
    return student_t.cdf(z * scale, df=nu)

def standardized_t_ppf(u, nu):
    """Inverse CDF of standardized Student-t."""
    scale = np.sqrt(nu / (nu - 2))
    return student_t.ppf(u, df=nu) / scale


def regime_count(start, end, label):
    dates = returns.index[regime_seq == label]
    return sum(start <= d.strftime('%Y-%m-%d') <= end for d in dates)

def save_regime_csv(dates, regime_seq, regime_probs, output_path,
                    regime_names=('Bull', 'Bear', 'SevereBear', 'Crisis')):
    """
    Save HMM regime labels and probabilities to CSV.

    Parameters
    ----------
    dates : array-like of datetime, length N
        Trading dates aligned with regime_seq.
    regime_seq : array-like of int, length N
        Hard regime label per date (0..3).
    regime_probs : 2D array, shape (N, 4)
        Soft regime probabilities per date.
    output_path : str or Path
        Where to write the CSV.
    regime_names : tuple of 4 str
        Names for the regime probability columns.

    Output columns:
        date, regime_seq, regime_prob_Bull, regime_prob_Bear,
        regime_prob_SevereBear, regime_prob_Crisis
    """
    import pandas as pd
    import numpy as np

    # Validate
    n = len(dates)
    if len(regime_seq) != n:
        raise ValueError(f"regime_seq length {len(regime_seq)} != dates length {n}")
    if regime_probs.shape != (n, 4):
        raise ValueError(f"regime_probs shape {regime_probs.shape} != ({n}, 4)")

    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'regime_seq': np.asarray(regime_seq, dtype=int),
        f'regime_prob_{regime_names[0]}': regime_probs[:, 0],
        f'regime_prob_{regime_names[1]}': regime_probs[:, 1],
        f'regime_prob_{regime_names[2]}': regime_probs[:, 2],
        f'regime_prob_{regime_names[3]}': regime_probs[:, 3],
    })
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return df
# =============================================================================
# 3. REGIME CLASSIFICATION VIA HMM
# =============================================================================
def fit_hmm_regimes(returns, n_regimes=N_REGIMES, feature_window=21):
    """
    Fit a Gaussian HMM on rolling regime features.
    Returns sequences aligned to the original `returns` index (length T).
    """
    print(f"\nFitting {n_regimes}-state HMM on rolling features...")
    T_full = len(returns)

    # Build features and drop initial NaN rows
    feats_full = make_regime_features(returns, window=feature_window)
    feats = feats_full.dropna()
    n_drop = T_full - len(feats)
    X = feats.values
    T_fit, _ = X.shape
    N = returns.shape[1]

    if HAS_HMMLEARN:
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=RANDOM_SEED,
            tol=1e-5,
        )
        model.fit(X)
        raw_seq_short = model.predict(X)
        raw_probs_short = model.predict_proba(X)
        trans_mat = model.transmat_

        # Compute means/covars on RETURNS (not features) for labeling
        ret_arr = returns.values
        # raw_seq_short corresponds to returns rows [n_drop:]
        means = np.array([
            ret_arr[n_drop:][raw_seq_short == k].mean(axis=0)
            if (raw_seq_short == k).sum() > 0 else np.zeros(N)
            for k in range(n_regimes)
        ])
        covars = np.array([
            np.cov(ret_arr[n_drop:][raw_seq_short == k].T)
            if (raw_seq_short == k).sum() > 1 else np.eye(N)
            for k in range(n_regimes)
        ])
    else:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=n_regimes, random_state=RANDOM_SEED, n_init=5)
        gm.fit(X)
        raw_seq_short = gm.predict(X)
        raw_probs_short = gm.predict_proba(X)

        ret_arr = returns.values
        means = np.array([
            ret_arr[n_drop:][raw_seq_short == k].mean(axis=0)
            if (raw_seq_short == k).sum() > 0 else np.zeros(N)
            for k in range(n_regimes)
        ])
        covars = np.array([
            np.cov(ret_arr[n_drop:][raw_seq_short == k].T)
            if (raw_seq_short == k).sum() > 1 else np.eye(N)
            for k in range(n_regimes)
        ])
        trans_mat = np.zeros((n_regimes, n_regimes))
        for t in range(T_fit - 1):
            trans_mat[raw_seq_short[t], raw_seq_short[t + 1]] += 1.0
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        trans_mat = np.divide(trans_mat, np.where(row_sums == 0, 1, row_sums))
        model = gm

    # Map raw HMM labels → canonical {0:Bull, 1:Bear, 2:Crisis, 3:InflationShock}
    remap = classify_regime_labels(means=means, covars=covars, assets=list(returns.columns))
    order = [raw for raw, canon in sorted(remap.items(), key=lambda kv: kv[1])]

    canon_seq_short = np.array([remap[r] for r in raw_seq_short], dtype=int)
    canon_probs_short = raw_probs_short[:, order]
    trans_sorted = trans_mat[np.ix_(order, order)]

    # Pad first n_drop rows: assign them to the same regime as the first valid day
    regime_seq = np.empty(T_full, dtype=int)
    regime_seq[n_drop:] = canon_seq_short
    regime_seq[:n_drop] = canon_seq_short[0]

    regime_probs = np.zeros((T_full, n_regimes))
    regime_probs[n_drop:] = canon_probs_short
    regime_probs[:n_drop] = canon_probs_short[0]   # broadcast first valid row

    # Diagnostics
    print("\nRegime diagnostics:")
    for k in range(n_regimes):
        mask = regime_seq == k
        cnt = int(mask.sum())
        if cnt == 0:
            print(f"  Regime {k} ({REGIME_NAMES[k]}): 0 observations")
            continue
        sub = returns.loc[mask]
        avg_v = sub.std(axis=0).values
        eq_mean = sub[["NVDA", "AMD", "SMH"]].mean().mean()
        tlt_mean = sub["TLT"].mean()
        print(
            f"  Regime {k} ({REGIME_NAMES[k]:14s}): {cnt:4d} days ({cnt/T_full*100:5.1f}%) | "
            f"eq_mean={eq_mean: .3f} | tlt_mean={tlt_mean: .3f} | "
            f"vol=[{', '.join(f'{v:.2f}' for v in avg_v)}]"
        )

    df_tm = pd.DataFrame(
        trans_sorted,
        index=[REGIME_NAMES[k] for k in range(n_regimes)],
        columns=[REGIME_NAMES[k] for k in range(n_regimes)],
    )
    print("\nTransition matrix (canonical regime order):")
    print(df_tm.round(3).to_string())

    # Save it
    with open(DATA_SYN_MOD / 'hmm_4regime.pkl', 'wb') as f:
        pickle.dump({
            'hmm': model,
            'feature_window': 21,  # the window you used for make_regime_features
            'regime_label_map': REGIME_NAMES,  # mapping from raw HMM states to canonical 0=Bull..3=Crisis
            'state_order':order,
        }, f)

    return regime_seq, regime_probs, model, trans_sorted


# =============================================================================
# 4. PER-REGIME GARCH + DCC + COPULA
# =============================================================================
def fit_constant_t(series):
    """Constant variance + Student-t innovations for small-sample regimes."""
    model = arch_model(series, vol="Constant", dist="StudentsT", mean="Constant")
    return model.fit(disp="off", show_warning=False)

def fit_garch_single(series, asset_name, min_obs_for_garch=1000):
    """
    Fit GARCH(1,1) with Student-t innovations.
    For small samples, fall back to constant-variance to avoid
    pathological parameter estimates from non-contiguous regime data.
    """
    if len(series) < min_obs_for_garch:
        # Constant variance + Student-t — GARCH dynamics not identifiable
        # on small disjoint samples
        model = arch_model(series, vol="Constant", dist="StudentsT", mean="Constant")
    else:
        model = arch_model(series, vol="Garch", p=1, q=1, dist="StudentsT", mean="Constant")
    return model.fit(disp="off", show_warning=False)


def fit_dcc_single(std_resids):
    """
    DCC(1,1) on standardized residuals.
    Returns:
        (a, b), R_series, Q_bar
    """
    Z = std_resids.values
    T, N = Z.shape
    Q_bar = np.cov(Z.T) if T > N else np.eye(N)

    def neg_ll(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Qt = Q_bar.copy()
        ll = 0.0
        for t in range(T):
            if t > 0:
                Qt = (1 - a - b) * Q_bar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Qt
            diag_inv = np.diag(1 / np.sqrt(np.diag(Qt)))
            Rt = diag_inv @ Qt @ diag_inv + np.eye(N) * 1e-6
            sign, ldet = np.linalg.slogdet(Rt)
            if sign <= 0:
                return 1e10
            zt = Z[t].reshape(-1, 1)
            quad = zt.T @ np.linalg.inv(Rt) @ zt  # shape (1, 1)
            ll += -0.5 * (ldet + quad.item())      # .item() works for any size-1 array, NumPy 1.x and 2.x
        return -ll

    res = minimize(
        neg_ll,
        x0=[0.03, 0.93],
        method="L-BFGS-B",
        bounds=[(1e-4, 0.2), (1e-4, 0.999)]
    )
    a, b = res.x

    Qt = Q_bar.copy()
    R_series = []
    for t in range(T):
        if t > 0:
            Qt = (1 - a - b) * Q_bar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Qt
        diag_inv = np.diag(1 / np.sqrt(np.diag(Qt)))
        Rt = diag_inv @ Qt @ diag_inv + np.eye(N) * 1e-6
        R_series.append(Rt)

    return (a, b), R_series, Q_bar


def fit_t_copula_single(U):
    N = U.shape[1]
    U_clip = np.clip(U.values, 1e-6, 1 - 1e-6)

    def neg_ll(nu_arr):
        nu = nu_arr[0]
        if nu <= 2:
            return 1e10
        X = student_t.ppf(U_clip, df=nu)
        R = np.corrcoef(X.T) + np.eye(N) * 1e-6
        sign, ldet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        Ri = np.linalg.inv(R)
        T_ = U_clip.shape[0]

        # Joint log density (vectorized)
        quad = np.einsum('ti,ij,tj->t', X, Ri, X)
        log_joint = (
                gammaln((nu + N) / 2)
                - gammaln(nu / 2)
                - (N / 2) * np.log(nu * np.pi)
                - 0.5 * ldet
                - ((nu + N) / 2) * np.log1p(quad / nu)
        )

        # Marginal log densities (sum across assets)
        log_marg = (
                gammaln((nu + 1) / 2)
                - gammaln(nu / 2)
                - 0.5 * np.log(nu * np.pi)
                - ((nu + 1) / 2) * np.log1p(X ** 2 / nu)
        ).sum(axis=1)

        return -(log_joint - log_marg).sum()

    res = minimize(neg_ll, [6.0], method="L-BFGS-B", bounds=[(2.1, 100)])
    nu = float(res.x[0])
    X = student_t.ppf(U_clip, df=nu)
    R = np.corrcoef(X.T)
    np.fill_diagonal(R, 1.0)
    return nu, R


def fit_per_regime(returns, volumes, regime_seq, n_regimes, assets, min_obs=60):
    """
    For each regime: fit per-asset GARCH + DCC + t-copula + volume model.
    """
    print("\nFitting per-regime GARCH + DCC + t-copula + volume models...")
    regime_models = []

    for k in range(n_regimes):
        name = REGIME_NAMES[k]
        mask = regime_seq == k
        obs = int(mask.sum())
        print(f"\n  Regime {k} ({name}) — {obs} observations")

        if obs < min_obs:
            print(f"    WARNING: too few observations ({obs}); placeholder will be filled later.")
            regime_models.append(None)
            continue

        ret_k = returns.loc[mask].copy()
        vol_k = volumes.loc[mask].copy()  # NEW
        ret_k.index = range(len(ret_k))

        # === GARCH fitting (unchanged) ===
        garch_k = {}
        std_res_k = pd.DataFrame()
        cond_vol_k = pd.DataFrame()

        for asset in assets:
            g = fit_garch_single(ret_k[asset], asset)
            garch_k[asset] = g
            std_res_k[asset] = g.std_resid.values
            cond_vol_k[asset] = g.conditional_volatility.values

            p = g.params
            if "omega" in p.index and "alpha[1]" in p.index:
                print(
                    f"    {asset}: ω={p['omega']:.4f} "
                    f"α={p['alpha[1]']:.4f} "
                    f"β={p['beta[1]']:.4f} "
                    f"ν={p['nu']:.2f}"
                )
            else:
                sigma = float(g.conditional_volatility.iloc[-1])
                print(f"    {asset}: σ={sigma:.4f} (constant) ν={p['nu']:.2f}")

        # === DCC fitting (unchanged, with constant-vol fallback) ===
        has_garch_dynamics = all("omega" in garch_k[a].params.index for a in assets)
        if has_garch_dynamics and obs >= 1000:
            dcc_params_k, R_series_k, Q_bar_k = fit_dcc_single(std_res_k)
            avg_R_k = np.mean(np.stack(R_series_k), axis=0)
            print(f"    DCC: a={dcc_params_k[0]:.4f} b={dcc_params_k[1]:.4f}")
        else:
            Q_bar_k = np.cov(std_res_k.values.T)
            R_series_k = [np.corrcoef(std_res_k.values.T)]
            avg_R_k = R_series_k[0]
            dcc_params_k = (0.0, 0.0)
            print(f"    DCC: skipped (constant variance regime, using static correlation)")

        # === Copula fitting (unchanged) ===
        U_k = pd.DataFrame({
            asset: standardized_t_cdf(std_res_k[asset].values, nu=garch_k[asset].params["nu"])
            for asset in assets
        })
        nu_cop_k, R_cop_k = fit_t_copula_single(U_k)
        print(f"    Copula ν={nu_cop_k:.2f}")

        last_h = {asset: float(cond_vol_k[asset].iloc[-1] ** 2) for asset in assets}

        eq_mean = float(ret_k[["NVDA", "AMD", "SMH"]].mean().mean())
        tlt_mean = float(ret_k["TLT"].mean())
        avg_vol = float(ret_k.std().mean())

        # === NEW: Volume model fitting per asset, attached INLINE ===
        regime_volume_model = {}
        last_log_vol = {}
        for asset in assets:
            vm = fit_volume_model_single(
                returns=ret_k[asset],
                volume=vol_k[asset].reset_index(drop=True),  # align indices
                asset_name=f"R{k}-{asset}",
            )
            regime_volume_model[asset] = vm
            last_log_vol[asset] = float(np.log(max(vol_k[asset].iloc[-1], 1.0)))
            print(
                f"    {asset} vol: α={vm['alpha']:.3f} β+={vm['beta_pos']:.4f} "
                f"β-={vm['beta_neg']:.4f} γ={vm['gamma']:.3f} σ={vm['sigma']:.3f}"
            )

        # === Append everything together (unchanged structure + new keys) ===
        regime_models.append({
            "name": name,
            "n_obs": obs,
            "garch": garch_k,
            "dcc_params": dcc_params_k,
            "Q_bar": Q_bar_k,
            "avg_R": avg_R_k,
            "R_series": R_series_k,
            "nu_copula": nu_cop_k,
            "R_copula": R_cop_k,
            "last_h": last_h,
            "mean_returns": ret_k.mean().values,
            "eq_mean": eq_mean,
            "tlt_mean": tlt_mean,
            "avg_vol": avg_vol,
            # NEW keys:
            "volume_model": regime_volume_model,
            "last_log_vol": last_log_vol,
        })

    # === Fill sparse regimes (unchanged) ===
    fill_model = next((m for m in regime_models if m is not None), None)
    if fill_model is None:
        raise RuntimeError("No regime had enough observations.")

    for i in range(len(regime_models)):
        if regime_models[i] is None:
            regime_models[i] = fill_model
            print(f"  Regime {i} ({REGIME_NAMES[i]}): filled from {fill_model['name']}")

    return regime_models


# =============================================================================
# 5. HYBRID SIMULATION
# =============================================================================
def simulate_hybrid_paths(
    regime_models,
    trans_mat,
    assets,
    initial_regime="random",
    n_steps=252,
    n_paths=500,
    stress_bias=None,
    return_prices=False,
    return_volumes=True,           # NEW
    start_price=100.0,
    seed=None,
):
    """
    Simulate synthetic paths using:
      1. Markov chain regime path
      2. regime-specific t-copula for correlated innovations
      3. regime-specific GARCH volatility propagation

    Parameters
    ----------
    stress_bias : dict or None
        Optional multipliers on regime transition probabilities.
        Example: {2: 1.5, 3: 1.5} to oversample Crisis / InflationShock.
    return_prices : bool
        If True, also returns reconstructed price paths.

    Returns
    -------
    all_returns : np.ndarray, shape (n_paths, n_steps, N), in %
    all_regimes : np.ndarray, shape (n_paths, n_steps), int regime labels
    all_prices  : np.ndarray, optional, shape (n_paths, n_steps+1, N)
    """
    rng = np.random.default_rng(seed)

    print(f"\nSimulating {n_paths} hybrid paths × {n_steps} steps...")
    N = len(assets)
    K = len(regime_models)
    all_returns = np.zeros((n_paths, n_steps, N), dtype=float)
    all_regimes = np.zeros((n_paths, n_steps), dtype=int)
    all_volumes = np.zeros((n_paths, n_steps, N), dtype=float)  # NEW

    for path in range(n_paths):
        if initial_regime == "random":
            regime = np.random.randint(K)
        else:
            regime = int(initial_regime)

        h = np.array([regime_models[regime]["last_h"][a] for a in assets], dtype=float)
        # NEW: initialize log-volume state from historical last value
        log_vol_state = np.array(
            [regime_models[regime]["last_log_vol"][a] for a in assets],
            dtype=float
        )

        for t in range(n_steps):
            rm = regime_models[regime]
            all_regimes[path, t] = regime

            # t-copula draw
            nu_c = rm["nu_copula"]
            R_c = rm["avg_R"] + np.eye(N) * 1e-8
            L = np.linalg.cholesky(R_c)
            z_n = L @ np.random.randn(N)
            chi2 = np.random.chisquare(nu_c) / nu_c
            z_t = z_n / np.sqrt(chi2)
            U_t = student_t.cdf(z_t, df=nu_c)

            ret = np.zeros(N, dtype=float)
            for i, asset in enumerate(assets):
                p = rm["garch"][asset].params
                mu = float(p["mu"])
                nu_g = float(p["nu"])
                innov = float(standardized_t_ppf(np.clip(U_t[i], 1e-6, 1 - 1e-6), nu=nu_g))

                if "omega" in p.index:
                    # GARCH dynamics
                    sig = np.sqrt(max(h[i], 1e-8))
                    ret[i] = mu + sig * innov
                    h[i] = (
                            float(p["omega"])
                            + float(p["alpha[1]"]) * (ret[i] - mu) ** 2
                            + float(p["beta[1]"]) * h[i]
                    )
                    h[i] = max(h[i], 1e-8)
                else:
                    # Constant variance — use stored last conditional vol
                    sigma_const = float(rm["garch"][asset].conditional_volatility.iloc[-1])
                    ret[i] = mu + sigma_const * innov
                    # h doesn't update; leave it at last_h value

            all_returns[path, t, :] = ret
            for i, asset in enumerate(assets):
                vm = rm["volume_model"][asset]
                log_vol_new, vol_new = simulate_volume_step(
                    ret_t=ret[i],
                    log_vol_prev=log_vol_state[i],
                    volume_model=vm,
                    rng=rng,
                )
                log_vol_state[i] = log_vol_new
                all_volumes[path, t, i] = vol_new


            # transition to next regime
            probs = np.array(trans_mat[regime], dtype=float).copy()
            if stress_bias is not None:
                for k, mult in stress_bias.items():
                    if 0 <= k < len(probs):
                        probs[k] *= float(mult)
                probs = probs / probs.sum()
            cum = np.cumsum(probs)
            regime = int(np.searchsorted(cum, np.random.rand()))
            regime = min(regime, K - 1)

    print(f"  Done. Returns shape: {all_returns.shape}, Volumes shape: {all_volumes.shape}")

    # Updated return logic
    out = [all_returns, all_regimes]
    if return_volumes:
        out.append(all_volumes)
    if return_prices:
        all_prices = build_synthetic_prices(all_returns, start_price=start_price)
        out.append(all_prices)
    return tuple(out) if len(out) > 2 else (all_returns, all_regimes)


# =============================================================================
# 6. SIMPLE DEMO RL ENVIRONMENT (NOT FINAL PROJECT ENV)
# =============================================================================
class SimpleRegimeAwareMarketEnv:
    """
    Simple demonstration environment only.

    Not intended as the final project environment because it is:
    - long-only
    - single-book
    - no 5-sleeve staggering
    - no HL/LL hierarchy
    - no shorting
    """

    def __init__(
        self,
        all_returns,
        all_regimes,
        tbill_daily,
        trans_mat,
        n_regimes=N_REGIMES,
        lookback=10,
        transaction_cost=0.001,
        sharpe_reward=False,
        sharpe_window=20
    ):
        self.returns = all_returns / 100.0
        self.regimes = all_regimes
        self.n_paths, self.n_steps, self.N = all_returns.shape
        self.tbill = tbill_daily.values if hasattr(tbill_daily, "values") else np.asarray(tbill_daily)
        self.trans_mat = trans_mat
        self.K = n_regimes
        self.lookback = lookback
        self.tc = transaction_cost
        self.sharpe_rew = sharpe_reward
        self.sw = sharpe_window
        self.n_assets_tot = self.N + 1
        self.reset()

    @property
    def obs_dim(self):
        return self.lookback * self.N + self.n_assets_tot + self.K + 1

    @property
    def action_dim(self):
        return self.n_assets_tot

    def reset(self, path_idx=None):
        self.path = path_idx if path_idx is not None else np.random.randint(self.n_paths)
        self.t = self.lookback
        self.weights = np.ones(self.n_assets_tot) / self.n_assets_tot
        self.beliefs = np.ones(self.K) / self.K
        self.ret_history = []
        return self._obs()

    def _update_beliefs(self, returns_t):
        predicted = self.trans_mat.T @ self.beliefs
        # simple soft observation update around actual latent regime (demo only)
        true_regime = self.regimes[self.path, self.t] if self.t < self.n_steps else 0
        one_hot = np.zeros(self.K)
        one_hot[true_regime] = 1.0
        updated = predicted * (0.7 * one_hot + 0.3 * np.ones(self.K) / self.K)
        s = updated.sum()
        self.beliefs = updated / s if s > 0 else np.ones(self.K) / self.K

    def _obs(self):
        hist = self.returns[self.path, self.t - self.lookback:self.t, :].flatten()
        rf = np.array([float(self.tbill[min(self.t, len(self.tbill) - 1)])])
        return np.concatenate([hist, self.weights, self.beliefs, rf])

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= action.sum() + 1e-8
        tc_cost = self.tc * np.sum(np.abs(action - self.weights))

        r_risky = self.returns[self.path, self.t, :]
        r_rf = float(self.tbill[min(self.t, len(self.tbill) - 1)])
        r_all = np.append(r_risky, r_rf)
        port_ret = float(np.dot(action, r_all)) - tc_cost

        self.ret_history.append(port_ret)
        self._update_beliefs(r_risky)
        self.weights = action
        self.t += 1
        done = self.t >= self.n_steps - 1

        if self.sharpe_rew and len(self.ret_history) >= self.sw:
            window = np.array(self.ret_history[-self.sw:])
            reward = window.mean() / (window.std() + 1e-8) * np.sqrt(252)
        else:
            reward = port_ret

        info = {
            "regime": int(self.regimes[self.path, self.t - 1]),
            "regime_name": REGIME_NAMES.get(int(self.regimes[self.path, self.t - 1]), "?"),
            "regime_beliefs": self.beliefs.copy(),
            "portfolio_return": port_ret,
            "transaction_cost": tc_cost,
        }
        return self._obs(), reward, done, info


# =============================================================================
# 7. DIAGNOSTICS
# =============================================================================
def plot_regime_overlay(returns, regime_seq, all_returns, all_regimes):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Regime-Switching + DCC-GARCH + t-Copula", fontsize=14, fontweight="bold")

    # historical portfolio return with shading
    ax = axes[0]
    portfolio_ret = returns.mean(axis=1)
    ax.plot(returns.index, portfolio_ret, color="#2c3e50", linewidth=0.6, alpha=0.9)

    for k in range(N_REGIMES):
        mask = regime_seq == k
        dates = returns.index[mask]
        for date in dates:
            ax.axvspan(date, date, alpha=0.12, color=REGIME_COLORS[k], linewidth=0)

    patches = [mpatches.Patch(color=REGIME_COLORS[k], alpha=0.5, label=REGIME_NAMES[k])
               for k in range(N_REGIMES)]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_title("Historical Portfolio Return with Regime Overlay")
    ax.set_ylabel("Log Return (%)")
    ax.grid(True, alpha=0.2)

    # smoothed regime probabilities from decoded labels
    ax = axes[1]
    T = len(regime_seq)
    probs = np.zeros((T, N_REGIMES))
    for k in range(N_REGIMES):
        probs[:, k] = pd.Series((regime_seq == k).astype(float)).rolling(21).mean().fillna(0).values
    bottom = np.zeros(T)
    for k in range(N_REGIMES):
        ax.fill_between(
            returns.index, bottom, bottom + probs[:, k],
            color=REGIME_COLORS[k], alpha=0.7, label=REGIME_NAMES[k]
        )
        bottom += probs[:, k]
    ax.set_ylim(0, 1)
    ax.set_title("Smoothed Regime Probabilities (21-day window)")
    ax.set_ylabel("P(regime)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    # synthetic path examples
    ax = axes[2]
    n_show = min(60, all_returns.shape[0])
    for i in range(n_show):
        starting_regime = int(all_regimes[i, 0])
        cum = np.cumprod(1.0 + all_returns[i, :, 0] / 100.0)
        ax.plot(cum, alpha=0.15, linewidth=0.5, color=REGIME_COLORS[starting_regime])

    patches = [mpatches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=f"Start: {REGIME_NAMES[k]}")
               for k in range(N_REGIMES)]
    ax.legend(handles=patches, fontsize=8)
    ax.set_title("Simulated NVDA Paths (colored by initial regime)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Normalized Price")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_regime_correlations(regime_models, assets):
    K = len(regime_models)
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4))
    fig.suptitle("Average DCC Correlation per Regime", fontsize=13, fontweight="bold")

    if K == 1:
        axes = [axes]

    for k, rm in enumerate(regime_models):
        ax = axes[k]
        R = pd.DataFrame(rm["avg_R"], index=assets, columns=assets)
        sns.heatmap(
            R, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True, cbar=(k == K - 1)
        )
        ax.set_title(
            f"Regime {k}: {rm['name']}\n"
            f"(n={rm['n_obs']} days, ν={rm['nu_copula']:.1f})",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


def plot_return_distributions(returns, all_returns, assets):
    N = len(assets)
    fig, axes = plt.subplots(2, N, figsize=(14, 8))
    fig.suptitle("Historical vs Simulated Return Distributions", fontsize=13, fontweight="bold")

    for i, asset in enumerate(assets):
        hist_ret = returns[asset].values
        sim_ret = all_returns[:, :, i].flatten()

        ax = axes[0, i]
        ax.hist(hist_ret, bins=80, density=True, alpha=0.6, color="#3498db", label="Historical")
        ax.hist(sim_ret, bins=80, density=True, alpha=0.5, color="#e74c3c", label="Simulated")
        ax.set_title(f"{asset} return dist.")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        ax2 = axes[1, i]
        h_s = np.sort(hist_ret)
        s_q = np.quantile(sim_ret, np.linspace(0, 1, len(h_s)))
        ax2.scatter(h_s, s_q, s=2, alpha=0.3, color="#9b59b6")
        lim = max(abs(h_s).max(), abs(s_q).max())
        ax2.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
        ax2.set_xlabel("Historical quantiles")
        ax2.set_ylabel("Simulated quantiles")
        ax2.set_title(f"{asset} Q-Q plot")
        ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def print_regime_summary(regime_models, trans_mat):
    print("\n" + "=" * 70)
    print("REGIME MODEL SUMMARY")
    print("=" * 70)
    for k, rm in enumerate(regime_models):
        print(f"\n{'─' * 50}")
        print(f"  Regime {k}: {rm['name']}  ({rm['n_obs']} days)")
        print(f"  eq_mean={rm['eq_mean']:.4f} | tlt_mean={rm['tlt_mean']:.4f} | avg_vol={rm['avg_vol']:.4f}")
        print(f"  Copula ν = {rm['nu_copula']:.2f}")
        print(f"  DCC: a={rm['dcc_params'][0]:.4f}  b={rm['dcc_params'][1]:.4f}")
        print("  Avg correlation matrix:")
        print(pd.DataFrame(rm["avg_R"], index=ASSETS, columns=ASSETS).round(3).to_string())

    print(f"\n{'─' * 50}")
    print("  Transition matrix:")
    df_t = pd.DataFrame(
        trans_mat,
        index=[REGIME_NAMES[k] for k in range(N_REGIMES)],
        columns=[REGIME_NAMES[k] for k in range(N_REGIMES)],
    )
    print(df_t.round(4).to_string())

    expected_durations = {}
    for k in range(N_REGIMES):
        pkk = min(max(trans_mat[k, k], 1e-6), 0.999999)
        expected_durations[REGIME_NAMES[k]] = round(1.0 / (1.0 - pkk), 2)
    print(f"\n  Approx. expected regime durations (days): {expected_durations}")
    print("=" * 70)


# =============================================================================
# 8. MAIN
# =============================================================================
def main():

    def rcount(start, end, label):
        dates = returns.index[regime_seq == label]
        return sum(start <= d.strftime('%Y-%m-%d') <= end for d in dates)

    prices, returns, tbill_daily = download_data(ASSETS, TBILL_TICKER, START, END)
    print(f"[line X] returns NaN={returns.isna().sum().sum()}, "
          f"shape={returns.shape}, dtype={returns.dtypes.unique()}")
    if not np.isfinite(returns).all().all():
        bad_mask = ~np.isfinite(returns)
        bad_rows = returns[bad_mask.any(axis=1)]  # rows with ≥1 non-finite value
        print(f"Found {bad_mask.sum().sum()} non-finite values in {len(bad_rows)} row(s):")
        print(bad_rows)
        exit()
    # Load volumes aligned to the same dates
    volumes = load_volumes_from_csvs(data_dir=DATA_DIR, assets=ASSETS,dates_index=returns.index)

    regime_seq, regime_probs, hmm_model, trans_mat = fit_hmm_regimes(returns, N_REGIMES)

    # Optional: empirical diagnostics
    diagnose_volume_relationships(returns, volumes, regime_seq, ASSETS)

    # Sanity: Crisis should fire during 2008-Q4 and 2020-Q1
    crisis_dates = returns.index[regime_seq == 2]
    print(f"\nCrisis days in 2008-Q4: {sum(('2008-10-01' <= d.strftime('%Y-%m-%d') <= '2008-12-31') for d in crisis_dates)}")
    print(f"Crisis days in 2020-Mar: {sum(('2020-03-01' <= d.strftime('%Y-%m-%d') <= '2020-04-15') for d in crisis_dates)}")

    print("Crisis regime — actual inflation episodes:")
    print(f"  2021-Q4 (CPI rising):        {rcount('2021-10-01', '2021-12-31', 3)} days")
    print(f"  2022-Q1 (war + inflation):   {rcount('2022-01-01', '2022-03-31', 3)} days")
    print(f"  2022-full year:              {rcount('2022-01-01', '2022-12-31', 3)} days")

    print("\nCrisis regime — non-inflation selloffs:")
    print(f"  2015-08 (China devaluation): {rcount('2015-08-01', '2015-09-30', 3)} days")
    print(f"  2018-Q4 (Fed selloff):       {rcount('2018-10-01', '2018-12-31', 3)} days")
    print(f"  2008-09 to 2009-06 (GFC):    {rcount('2008-09-01', '2009-06-30', 3)} days")
    print(f"  2020-02 to 2020-04 (COVID):  {rcount('2020-02-15', '2020-04-30', 3)} days")
    print(f"  2011-08 (US debt ceiling):   {rcount('2011-08-01', '2011-08-31', 3)} days")
    print(f"  2018-12 (Q4 selloff):        {rcount('2018-10-01', '2018-12-31', 3)} days")

    print("Severe Bear dates check:")
    print(f"  2008-09 to 2009-06 (GFC):       {rcount('2008-09-01', '2009-06-30', 2)} days")
    print(f"  2020-02 to 2020-04 (COVID):     {rcount('2020-02-15', '2020-04-30', 2)} days")
    print(f"  2011-08 (US debt ceiling):      {rcount('2011-08-01', '2011-08-31', 2)} days")
    print(f"  2018-12 (Q4 selloff):           {rcount('2018-10-01', '2018-12-31', 2)} days")

    print("\nBull dates check:")
    print(f"  2017 (low-vol bull year):       {rcount('2017-01-01', '2017-12-31', 0)} days")
    print(f"  2013 (taper tantrum aside):     {rcount('2013-01-01', '2013-12-31', 0)} days")


    print("\nRegime summary table:")
    print(regime_summary_table(regime_seq, returns).round(4).to_string(index=False))

    regime_models = fit_per_regime(returns, volumes, regime_seq, N_REGIMES, ASSETS)

    # Save the FITTED MODELS once (no seed needed)
    dates = returns.index
    fitted_parameters = {
            'regime_models': regime_models,
            'trans_mat': trans_mat,
            'regime_seq': regime_seq,
            'regime_probs': regime_probs,
            'training_dates': dates,
    }

    save_regime_csv(dates=dates, regime_seq=regime_seq, regime_probs=regime_probs, output_path=DATA_SYN_MOD / 'regime_labels.csv')



    with open( DATA_SYN_MOD / 'synthetic_generator_FITTED.pkl', 'wb') as f:
        pickle.dump(fitted_parameters, f)


    print_regime_summary(regime_models, trans_mat)

    all_returns, all_regimes, all_volumes, all_prices = simulate_hybrid_paths(
        regime_models=regime_models,
        trans_mat=trans_mat,
        assets=ASSETS,
        initial_regime="random",
        n_steps=252,
        n_paths=500,
        return_volumes=True,
        seed=42,
        stress_bias=None,
        return_prices=True,
        start_price=100.0,
    )

    env = SimpleRegimeAwareMarketEnv(
        all_returns=all_returns,
        all_regimes=all_regimes,
        tbill_daily=tbill_daily,
        trans_mat=trans_mat,
        n_regimes=N_REGIMES,
        lookback=10,
        transaction_cost=0.001,
        sharpe_reward=False,
    )

    print("\nSimple RL Demo Environment:")
    print(f"  obs_dim    = {env.obs_dim}")
    print(f"  action_dim = {env.action_dim}")
    print(f"  n_paths    = {env.n_paths}")
    print(f"  n_steps    = {env.n_steps}")

    # quick sanity with a random agent
    obs = env.reset()
    total_ret = 0.0
    regime_counts = np.zeros(N_REGIMES, dtype=int)
    done = False
    while not done:
        action = np.random.dirichlet(np.ones(env.action_dim))
        obs, reward, done, info = env.step(action)
        total_ret += info["portfolio_return"]
        regime_counts[info["regime"]] += 1

    print(f"\n  Random agent episode return: {total_ret * 100:.2f}%")
    print("  Regime distribution in episode: " + ", ".join(
        [f"{REGIME_NAMES[k]}={int(regime_counts[k])}" for k in range(N_REGIMES)]
    ))

    # optional plots
    plot_regime_overlay(returns, regime_seq, all_returns, all_regimes)
    plot_regime_correlations(regime_models, ASSETS)
    plot_return_distributions(returns, all_returns, ASSETS)

    return {
        "prices": prices,
        "returns": returns,
        "tbill_daily": tbill_daily,
        "regime_seq": regime_seq,
        "regime_probs": regime_probs,
        "regime_models": regime_models,
        "trans_mat": trans_mat,
        "all_returns": all_returns,
        "all_regimes": all_regimes,
        "all_prices": all_prices,
        "env": env,
    }


if __name__ == "__main__":
    outputs = main()
    print("\n\n" + "=" * 70)
    print('=> OUTPUTS:')
    print(outputs)
