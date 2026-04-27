from copy import deepcopy
from IB_Data import IB_Data as ib
from IB_Data import Technicals_TimeSeries as tech
from IB_Data import Technicals_Wavelets as wave
from datetime import date

# This uses Numa proprietary Library to fetch most recent prices and calculates selected technicals indicators.
semi = ['NVDA', 'AMD', 'SMH', 'TLT']
data = ib.readDownloadedPricesFiles(['SemiConductors', 'TLT'], semi, update=False)


technicals = tech(data)

# ---- Feature generation (these only use historical bars)
technicals.volume()
technicals.ml_safe_technicals()
for ticker in semi:
    print(f'Column length of {ticker} is {len(technicals.data['NVDA'].columns)} before Kronos')

# ---- IMPORTANT: Kronos must come BEFORE regimes
technicals.integrate_kronos_multi_v3()
for ticker in semi:
    print(f'Column length of {ticker} is {len(technicals.data['NVDA'].columns)} after Kronos')

#---- Column ordering and filtering
technicals.data = ib.arrangeColumns(technicals.data, ['date'])
# ---- IMPORTANT: Wavelets
wavelets = wave(technicals)
wavelets.run_pipeline()
wavelets.run_pipeline(wavelet_window=90)
wavelets.compute_cross_scale_features(90, 128)

# ---- Export pipeline
# ---- Drop unwanted raw columns AFTER wwavelet feature creation
dropCol = ['ret1', 'open', 'high', 'low', 'VWAP', 'Volume', 'V_raw']
wavelets.data = ib.drop(wavelets.data, dropCol)
# ---- Check feature Length
for ticker in semi:
    print(f'Column length of {ticker} is {len(wavelets.data[ticker].columns)} FINAL')

# Eliminates dates that have NaN and shortens data to shortest data which is TLT
ib.getFirstLastDate(wavelets.data)
cleanData = deepcopy(wavelets)
cleanData.data= ib.dropDataAboveBelowDate(cleanData.data, above=False, dropD=date(year=2004, month=8, day=12), fieldDate='date')
ib.getFirstLastDate(cleanData.data, ' Clean Data After dropping dates below 2004-08-12')

# Train
trainData = deepcopy(cleanData)
trainData= ib.dropDataAboveBelowDate(trainData.data, dropD=date(year=2022, month=12, day=31), fieldDate='date')
ib.checkNaN(trainData)
ib.getFirstLastDate(trainData, msg=' Train Data After dropping dates above 2022-12-31')

#Test
testData = deepcopy(cleanData)
testData= ib.dropDataAboveBelowDate(testData.data, above=False, dropD=date(year=2022, month=12, day=31), fieldDate='date')
#kronos last prediction
testData= ib.dropDataAboveBelowDate(testData, above=True, dropD=date(year=2026, month=4, day=23), fieldDate='date')
ib.checkNaN(testData)
ib.getFirstLastDate(testData, msg=' Test Data between  2022-12-31 and 20026-04-24')

# Save real data as csv
ib.toCSV(testData, 'RL_RealData_Individual_Features_Test', ndx=False)
ib.toCSV(trainData, 'RL_RealData_Individual_Features_Train', ndx=False)

# Save Data pickle files are in the format dictionary of {ticker}:df
ib.to_Picke(cleanData, 'clean_data_features', theDir=ib.downloadPath())
ib.to_Picke(trainData, 'train_data_features', theDir=ib.downloadPath())
ib.to_Picke(testData, 'test_data_features', theDir=ib.downloadPath())