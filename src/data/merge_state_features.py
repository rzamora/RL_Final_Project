import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")
import pickle
import sys

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "proccessed" / "individual_securities"
DATA_TRAIN = DATA_DIR / "train_real"
DATA_TRAIN_PCK = DATA_TRAIN / "picke"
TBILL_TICKER = "DTB3"
START        = "2004-08-13"  # TLT has the least history, this is the start date of TLT
END          = "2022-12-30"  # set so we can test 2023, 2024, 2025 and 2026

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def getFirstLastDate(theDF, msg='', verbose=True):
    '''
    :return first and last date with time
    '''

    def processDf(theDF):
        if len(theDF) > 0:
            firstDate = theDF.index[0]
            lastDate = theDF.index[-1]
            if verbose:
                if len(theDF.TICKER.unique()) > 1:
                    try:
                        dTmeM = IB_Data.getRowBarSize(theDF).total_seconds() / 60
                    except:
                        print('ERROR in calculating barsize.. assuming 1 day. ' + msg)
                        dTmeM = 24 * 60
                else:
                    dTmeM = (theDF.index[1] - theDF.index[0]).total_seconds() / 60
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
                print('IB DTA: ' + msg + ' Time Between Rows {0} minutes or {1:.3f} hours '.format(dTmeM, dTmeH), '\n')
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
    data = get_Picke(os.path.join(DATA_TRAIN_PCK, 'train_data_features'))
    ASSETS = data.keys()
    print(f'Length of data is {len(data)} with tickers: {ASSETS}')
    getFirstLastDate(data)



if __name__ == "__main__":
    outputs = main()
