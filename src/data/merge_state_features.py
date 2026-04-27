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
# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "proccessed" / "individual_securities"
DATA_MERGED = PROJECT_ROOT / "data" / "proccessed" / "combined_w_cross_asset"
DATA_DIR_OUT = PROJECT_ROOT / "data" / "proccessed" / "individual_securities/all"
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
# 10. MAIN
# =============================================================================
def main():
    data = get_Picke(os.path.join(DATA_ALL_PCK, 'all_data_features'))
    ASSETS = data.keys()
    print(f'Length of data is {len(data)} with tickers: {ASSETS}')
    data = dropDataAboveBelowDate(data, dropD=date(year=2026, month=4, day=23), above=True)
    getFirstLastDate(data)



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

    # Step 4: Trim all assets
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
    test = dropDataAboveBelowDate(deepcopy(merged), dropD=date(year=2022, month=12, day=31), above=False)
    # Step 5: Save
    merged.to_csv(DATA_MERGED / 'RL_Final_Merged_All.csv', index=True)
    train.to_csv(DATA_MERGED / 'train' / 'RL_Final_Merged_train.csv', index=True)
    test.to_csv(DATA_MERGED / 'test' / 'RL_Final_Merged_test.csv', index=True)


if __name__ == "__main__":
    outputs = main()
