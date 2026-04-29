from copy import deepcopy

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")
import pickle
import sys
from datetime import datetime, date, time
from cross_asset_correlations import add_correlation_features, diagnose_correlation_features
from merge_asset_features_into_one import merge_asset_features
from regime_dcc_garch_copula_V1 import make_regime_features
# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "proccessed" / "individual_securities"
DATA_MERGED = PROJECT_ROOT / "data" / "proccessed" / "combined_w_cross_asset"
DATA_DIR_OUT = PROJECT_ROOT / "data" / "proccessed" / "individual_securities/all"
DATA_SYN_MOD = PROJECT_ROOT / "data" / "synthetic" / "models"
DATA_TRAIN = DATA_DIR / "train_real"
DATA_TRAIN_PCK = DATA_TRAIN / "picke"
DATA_ALL_PCK = DATA_DIR / "picke"
TBILL_TICKER = "DTB3"
START        = "2004-08-13"  # TLT has the least history, this is the start date of TLT
END          = "2022-12-30"  # set so we can test 2023, 2024, 2025 and 2026

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def getRowBarSize(theDF, msg='', timedelta=True):
    '''
    :param theDF:  Assumes the dataframe is indexed by Date amd is datetime type
    :return bar size of the rows
    '''
    theDF = theDF.loc[theDF.TICKER == theDF.TICKER.unique()[0]]
    barSize = theDF.index[1] - theDF.index[0]
    if not timedelta:
        barSize = barSize.total_seconds()
    print('IB DTA: ' + msg + ' The Bar Size of the rows is ', barSize, ' seconds.')
    return barSize

def checkNaN(data, show_locations=True, show_dates=True, verbose=False):
    def _diagnose_nan_position(df, col):
        """Classify where a column's NaN values are located by integer position."""
        is_nan = df[col].isna().values  # NumPy bool array
        if not is_nan.any():
            return '', None, None

        # Get integer positions, not index labels
        nan_positions = np.where(is_nan)[0]
        first_pos = int(nan_positions.min())
        last_pos = int(nan_positions.max())
        n = len(df)

        head_count = int((nan_positions < n * 0.1).sum())
        tail_count = int((nan_positions > n * 0.9).sum())
        mid_count = len(nan_positions) - head_count - tail_count

        labels = []
        if head_count > 0:
            labels.append(f"head:{head_count}")
        if mid_count > 0:
            labels.append(f"middle:{mid_count}")
        if tail_count > 0:
            labels.append(f"tail:{tail_count}")

        return ' '.join(labels), first_pos, last_pos

    def _check_one(df, name=''):
        feat_cols = [c for c in df.columns if c not in ['index', 'date', 'DATE', 'TICKER']]
        n_nan = df[feat_cols].isna().sum().sum()
        prefix = f"{name}: " if name else ""
        print(f"{prefix}{len(df)} rows, {n_nan} remaining NaN")

        if n_nan == 0:
            return

        nan_counts = df[feat_cols].isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

        date_col = 'date' if 'date' in df.columns else 'DATE' if 'DATE' in df.columns else None

        if not show_locations:
            print(nan_counts.to_string())
            return

        # Header
        if show_dates and date_col is not None:
            print(
                f"  {'Column':<35s} {'Count':>6s}  {'Position':<25s}  {'First NaN date':<14s} {'Last NaN date':<14s}")
        else:
            print(f"  {'Column':<35s} {'Count':>6s}  {'Position':<25s}")

        for col, count in nan_counts.items():
            position, first_idx, last_idx = _diagnose_nan_position(df, col)
            if show_dates and date_col is not None:
                first_date = pd.to_datetime(df.loc[first_idx, date_col]).date()
                last_date = pd.to_datetime(df.loc[last_idx, date_col]).date()
                print(f"  {col:<35s} {count:>6d}  {position:<25s}  {str(first_date):<14s} {str(last_date):<14s}")
            else:
                print(f"  {col:<35s} {count:>6d}  {position:<25s}")

    if isinstance(data, dict):
        for ticker, df in data.items():
            feat_cols = [c for c in df.columns if c not in ['index', 'date', 'DATE', 'TICKER']]
            n_nan = df[feat_cols].isna().sum().sum()
            print(f"{ticker}: {len(df)} rows, {n_nan} remaining NaN")
            if n_nan > 0:
                nan_counts = df.isna().sum()
                nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
                print(nan_cols)
                _check_one(df, ticker)
    elif isinstance(data, pd.DataFrame):
        feat_cols = [c for c in data.columns if c not in ['index', 'date', 'DATE', 'TICKER']]
        n_nan = data[feat_cols].isna().sum().sum()
        print(f"{len(data)} rows, {n_nan} remaining NaN")
        if n_nan > 0:
            _check_one(data)
    return data

def dropDataAboveBelowDate(data, dropD=date(year=2019, month=12, day=31), above=True, verbose=True, fieldDate="date"):
    '''
    :param dropD:  Assumes the dataframe is indexed by Date
    :return:
    '''
    if isinstance(data, dict):
        data = data.copy()
        for ticker, df in data.items():
            originalLength = len(df)
            if verbose:
                print('Data length before dropping ', ticker, ' rows are ', len(df))
            dropDate = datetime.combine(dropD, time.min)
            if fieldDate in df.columns:
                setIndxDate = False
            else:
                setIndxDate = True
                df.reset_index(inplace=True)
            if above:
                df = df[df[fieldDate] <= dropDate]
            else:
                df = df[df[fieldDate] > dropDate]
            if setIndxDate:
                df = df.set_index(fieldDate)
                getFirstLastDate(df, verbose=verbose)
            if verbose:
                print('TSD :Dropped', originalLength - len(df), ' for ticker ', ticker)
            data[ticker] = df
        return data
    elif isinstance(data, pd.DataFrame):
        data = data.copy()
        originalLength = len(data)
        if verbose:
            print('TSD :Data length before dropping Above / Below Date ', originalLength)
        if fieldDate in data.columns:
            setIndxDate = False
            print('TSD :Date column is in the dataframe')
        else:
            setIndxDate = True
            data.reset_index(inplace=True)
            print('TSD :Date column is not in the dataframe')
        dropDate = datetime.combine(dropD, time.min)
        if above:
            data = data[data[fieldDate] <= dropDate]
        else:
            data = data[data[fieldDate] > dropDate]
        data = data.set_index(fieldDate)
        getFirstLastDate(data, verbose=verbose)
        if verbose:
            print('TSD :Dropped', originalLength - len(data))
        return data
    else:
        print('\nDROP:Error variable data must be a dictionary of ticker:dataframe or single dataframe', type(data))



def getFirstLastDate(theDF, msg='', verbose=True):
    '''
    :return first and last date with time
    '''

    def processDf(theDF):
        if len(theDF) > 0:
            firstDate = theDF.index[0]
            lastDate = theDF.index[-1]
            if verbose:
                dTmeM = (theDF.index[-1] - theDF.index[-2]).total_seconds() / 60
                dTmeH = dTmeM / 60
                dTmeD = dTmeH / 24
                if dTmeD < 1:
                    f = firstDate
                    l = lastDate
                else:
                    f = firstDate.date()
                    l = lastDate.date()
                print('IB DTA: ' + msg + ' The start Date is {}'.format(f))
                print('IB DTA: ' + msg + ' The end date is ', l)
                print('IB DTA: ' + msg + ' Total Number of days = ', (lastDate - firstDate).days)
                print('IB DTA: ' + msg + ' Total Number of Rows = ', len(theDF))
            return firstDate, lastDate
        else:
            print('\nIB DTA: ' + msg + ' Dataframe has no rows\n')

    if isinstance(theDF, dict):
        for ticker, df in theDF.items():
            print(f'proccesing {ticker} dataframe for start and end date')
            if 'date' in df.columns: df.set_index('date', inplace=True)
            if 'DATE' in df.columns: df.set_index('DATE', inplace=True)
            processDf(df)
    elif isinstance(theDF, pd.DataFrame):
        processDf(theDF)


def get_Picke(filename):
    try:
        with open(filename, 'rb') as f:
            theData = pickle.load(f)
            print('IB DTA: Pickle file is loaded successfully')
            return theData

    except Exception as e:
        print('DTA: Cannot find file to get pickle or there is an error', filename)
        print('The exception is ', str(e))
        sys.exit(1)


def downloadPath():
    return os.path.join(os.path.join(os.path.expanduser('~')), 'Downloads/')

def desktopPath():
    return os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

def mergeColumns(data, dfAdd, tickerAdd):
    if isinstance(data, dict):
        for ticker, df in data.items():
            data[ticker] = df.merge(dfAdd[tickerAdd], left_index=True, right_index=True)
            print('Length of ', ticker, ' column features is ', len(data[ticker].columns))
    else:
        print('\nDROP:Error variable data must be a dictionary of ticker:dataframe', type(data))
    return data
# =============================================================================
# 10. PREDICT REGIME PROBs for TEST SET
# =============================================================================

def predict_test_regime_probs(test, train):
    '''
    This function predicts the regime probabilities for the test set using the fitted HMM.
    '''

    aux_df = deepcopy(test)
    # Load fitted HMM
    with open(DATA_SYN_MOD /'hmm_4regime.pkl', 'rb') as f:
        saved = pickle.load(f)
    model = saved['hmm']
    feature_window = saved['feature_window']
    state_order = saved['state_order']  # e.g. [2, 0, 3, 1] meaning raw state 2 = canonical Bull
    label_map = saved['regime_label_map']

    warmup = train.iloc[-feature_window:]  # last 21 trading days of train
    returnCols = [c for c in warmup.columns if c.endswith('_CPct_Chg1')]
    warmup = warmup[returnCols]
    warmup.columns = [c.split('_')[0] for c in warmup.columns]

    # Load test data — must have the same return columns as training data
    # test_returns = pd.read_csv('data/test_returns.csv', parse_dates=['date'], index_col='date')
    returnCols = [c for c in aux_df.columns if c.endswith('_CPct_Chg1')]
    aux_df = aux_df[returnCols]
    aux_df.columns = [c.split('_')[0] for c in aux_df.columns]

    # Compute the SAME rolling features used during training
    combined = pd.concat([warmup, aux_df])
    combined_features = make_regime_features(combined, window=feature_window)
    test_features = combined_features.iloc[feature_window:]  # drop the warmup rows
    print(f"sum first 3 rows if isna :{test_features.head(3).isna().sum()} all zeroes expected")
    print(f"Length: {len(test_features)} (should equal len(test) = {len(test)})")

    # Drop any rows where features are NaN (the warmup window)
    valid_mask = test_features.notna().all(axis=1)
    test_features_valid = test_features[valid_mask].values
    print(f'Number of valid test features: {len(test_features_valid)} vs {len(test_features)}')

    # -------------------------------------------------------------------------
    # Predict regime probabilities on test data — strictly causal (filtered)
    # -------------------------------------------------------------------------
    # At each step t, we call predict_proba on the truncated history [:t+1] and
    # take only the last row. Because there is nothing past index t in the
    # truncated sequence, hmmlearn's backward pass is a no-op at the boundary
    # and the smoothed posterior gamma_t collapses to the filtered posterior:
    #       P(state_t | x_1, ..., x_t)
    # This is the strictly causal quantity an online RL agent could observe at
    # decision time t — no future information leaks into earlier labels.
    #
    # The loop is O(n^2) but n=829 here and total runtime is a few seconds.
    # An O(n) equivalent is available via hmmlearn's internal forward pass
    # (_compute_log_likelihood + _do_forward_pass) but the public API is more
    # stable and the numerical output is identical.
    probs_online = []
    for t in range(len(test_features_valid)):
        history = test_features_valid[:t + 1]
        probs_history = model.predict_proba(history)
        probs_online.append(probs_history[-1])  # filtered posterior at time t
    probs_online = np.array(probs_online)

    if len(probs_online) != len(test):
        raise RuntimeError(
            f"probs_online length {len(probs_online)} does not equal len(test) "
            f"{len(test)}. Cannot align regime probabilities to test rows."
        )

    # -------------------------------------------------------------------------
    # CRITICAL: Reorder raw HMM state columns -> canonical {Bull, Bear, SB, Crisis}
    # -------------------------------------------------------------------------
    # `model.predict_proba` returns columns indexed by raw HMM state (arbitrary,
    # determined by EM init). The canonical order [Bull, Bear, SevereBear, Crisis]
    # was established at fit time by classify_regime_labels() and stored in the
    # `state_order` array: state_order[c] is the raw HMM state index that
    # corresponds to canonical class c.
    #
    # This reorder was applied to the train probs at fit time (see
    # fit_hmm_regimes -> canon_probs_short = raw_probs_short[:, order]) but was
    # previously omitted on test, so test labels were a permutation of the
    # canonical names. That permutation explains the train/test inversion seen
    # in the regime classifier diagnostic.
    #
    # Requires fit_hmm_regimes() to save 'state_order' into the pickle. See the
    # matching patch in regime_dcc_garch_copula_V1.py.
    state_order = saved.get('state_order')
    if state_order is None:
        raise KeyError(
            "Pickle missing 'state_order' — cannot map raw HMM states to canonical "
            "regime names. Re-fit the HMM with fit_hmm_regimes() updated to save "
            "'state_order' alongside 'hmm' and 'regime_label_map'."
        )

    probs_canonical = probs_online[:, state_order]

    # Sanity: rows should still sum to 1 (reordering is a column permutation)
    row_sums = probs_canonical.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"Canonical probs do not sum to 1 (range: {row_sums.min():.6f} - "
            f"{row_sums.max():.6f}). Likely an issue with state_order."
        )

    # -------------------------------------------------------------------------
    # Write canonical-order probs into the test DataFrame
    # -------------------------------------------------------------------------
    prob_columns = ['regime_prob_Bull', 'regime_prob_Bear',
                    'regime_prob_SevereBear', 'regime_prob_Crisis']
    test[prob_columns] = probs_canonical

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    test_regime_seq = np.argmax(probs_canonical, axis=1)
    regime_names = ['Bull', 'Bear', 'SevereBear', 'Crisis']

    print("\nRaw HMM state -> canonical regime mapping (from training):")
    for canon_idx, raw_idx in enumerate(state_order):
        print(f"  raw HMM state {raw_idx}  ->  canonical {canon_idx} ({regime_names[canon_idx]})")

    print("\nTest regime distribution (canonical labels, filtered/causal):")
    seen = dict(zip(*np.unique(test_regime_seq, return_counts=True)))
    total = len(test_regime_seq)
    for c, name in enumerate(regime_names):
        n = int(seen.get(c, 0))
        print(f"  {name:<12s}  {n:>4d} days  ({n / total * 100:>5.1f}%)")

    print("Saving the test file with regime probabilities...")
    test.to_csv(DATA_MERGED / 'test' / 'RL_Final_Merged_test.csv', index=True)
    return test_regime_seq

    # This reorders columns so column 0 = Bull, column 1 = Bear, etc.
# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    data = get_Picke(os.path.join(DATA_ALL_PCK, 'all_data_features'))
    ASSETS = data.keys()
    print(f'Length of data is {len(data)} with tickers: {ASSETS}')
    data = dropDataAboveBelowDate(data, dropD=date(year=2026, month=4, day=23), above=True)
    getFirstLastDate(data)

    # =============================================================================
    # CORRELATIONS
    # =============================================================================

    correlations = add_correlation_features(
        data,
        return_col='CPct_Chg1',
        date_col='date',
        windows=(20, 60),
        zscore_window=(30, 120),
        equity_assets=['NVDA', 'AMD', 'SMH'],
        tlt_asset='TLT',
    )
    correlations['TICKER'] = ''
    data['COR']=correlations

    #for ticker, df in enriched.items():
    correlations.to_csv(os.path.join(DATA_DIR_OUT, 'RL_Final_Group_Correlation.csv'), index=False)
    # After running add_correlation_features:
    # Step 1: Find common dates across all assets
    common_dates = set.intersection(*[set(df['date']) for df in data.values()])

    # Step 2: Trim to common dates
    for ticker in data:
        df = data[ticker]
        data[ticker] = df[df['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)

    # Step 3: Find the contiguous fully-clean window
    ref = data['COR']
    feat_cols = [c for c in ref.columns if c not in ['index', 'date', 'DATE', 'TICKER']]
    clean_mask = ref[feat_cols].notna().all(axis=1)
    clean_indices = ref.index[clean_mask]

    if len(clean_indices) == 0:
        raise ValueError("No fully-clean rows found!")

    first_clean = clean_indices.min()
    last_clean = clean_indices.max()
    clean_start_date = ref.loc[first_clean, 'date']
    clean_end_date = ref.loc[last_clean, 'date']

    print(f"Clean window: {clean_start_date.date()} → {clean_end_date.date()} "
          f"({last_clean - first_clean + 1} rows)")

    # Step 4: Trim all assets + Regime
    for ticker in data.keys():
        data[ticker] = data[ticker].iloc[first_clean:last_clean + 1].reset_index(drop=True)

    data = dropDataAboveBelowDate(data, dropD=date(year=2026, month=10, day=23), above=True)

    # Verify NAN
    checkNaN(data)

    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} rows, {df['date'].min().date()} → {df['date'].max().date()}")

    merged = merge_asset_features(data, asset_tickers=('NVDA', 'AMD', 'SMH', 'TLT'), cross_asset_ticker='COR',
        date_col='date',
        drop_cols=('TICKER', 'index'),
        save_to=os.path.join(DATA_DIR_OUT, 'RL_Final_Merged.csv'),
    )

    train = dropDataAboveBelowDate(deepcopy(merged), dropD=date(year=2022, month=12, day=31), above=True)
    # =============================================================================
    # REGIME PROBABILITIES
    # =============================================================================
    regime_df = pd.read_csv(DATA_SYN_MOD / 'regime_labels.csv', parse_dates=['date'])
    regime_df .drop(['regime_seq'], axis=1, inplace=True)
    train.reset_index(inplace=True)
    common_dates = set.intersection(*[set(df['date']) for df in [train, regime_df]])
    regime_df= regime_df[regime_df['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    print(f'Length of regime_df is {len(regime_df)} with lenght of train {len(train)}')
    reg_cols = [c for c in regime_df.columns if c != 'date']
    train = train.merge(regime_df[['date'] + reg_cols], on='date', how='left')
    train.set_index('date', inplace=True)

    test = dropDataAboveBelowDate(deepcopy(merged), dropD=date(year=2022, month=12, day=31), above=False)
    # Step 5: Save
    merged.to_csv(DATA_MERGED / 'RL_Final_Merged_All.csv', index=True)
    train.to_csv(DATA_MERGED / 'train' / 'RL_Final_Merged_train.csv', index=True)
    predict_test_regime_probs(test, train)  # will also save it to DATA_MERGED / 'test' / 'RL_Final_Merged_test.csv'



if __name__ == "__main__":
    outputs = main()
