from IB_Data import IB_Data as ib

semi = ['NVDA', 'AMD', 'SMH', 'TLT', 'DTB3']
data = ib.readDownloadedPricesFiles(['SemiConductors', 'TLT', 'DTB3'], semi, update=False)
'''
| ******** DOWNLOAD EACH FILE TO MAC OS DOWNLOAD FOLDER *******
Must have access to Numa Americas Directories in Dropbox
'''
data = ib.calculateLogReturns(data,  ['NVDA', 'AMD', 'SMH', 'TLT'])
data = ib.calculateRateReturns(data, ['DTB3'])
ib.toCSV(data, 'StockData_RL', ndx=False)
