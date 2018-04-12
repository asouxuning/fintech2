import tushare as ts
import numpy as np
import pandas as pd
import os
import time

stock_basics_path = 'data/stock_basics.csv'
if not os.path.exists(stock_basics_path):
  print('downloading...')
  stock_basics = ts.get_stock_basics()
  stock_basics.to_csv(stock_basics_path)

stock_basics = pd.read_csv(stock_basics_path, 
                  dtype={'code': str, 'timeToMarket': str})
stock_basics.index = stock_basics['code']
stock_basics.loc[:,'timeToMarket'] = pd.to_datetime(stock_basics['timeToMarket'], errors='coerce')
stock_type = 'k'
#stock_type = 'hist'

start = '1990'
i = 0
for code in stock_basics.index:
  
  store_path = 'data/'+stock_type+'/'+code+'.csv'
  if stock_type == 'k':
    stock_data = ts.get_k_data(code, start=start)
  else:
    stock_data = ts.get_hist_data(code, start=start)
  stock_data.to_csv(store_path)
  name = stock_basics.loc[code,'name']
  timeToMarket = stock_basics.loc[code,'timeToMarket']
  print("%d: %s(%s)\t-\t'%s'" % (i, name, code, timeToMarket))

  #if timeToMarket is not pd.to_datetime(np.nan):
  #  print(timeToMarket)
  #else:
  #  print('NaT')
  i += 1
