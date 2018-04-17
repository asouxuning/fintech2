import tushare as ts
import numpy as np
import pandas as pd
import os
import time

def update_stock_data(code, stock_type):
  path = 'data/'+stock_type+'/'+code+'.csv'
  
  x_cols = ['open','close','high','low','volume']
  stock_data = pd.read_csv(path)
  stock_data.index = pd.to_datetime(stock_data['date'])
  stock_data = stock_data.sort_index(ascending=True)
  stock_data = stock_data[x_cols]
  
  last_date = stock_data.index[-1]
  new_tm = last_date.timestamp() + 60*60*24
  new_date = last_date.fromtimestamp(new_tm)
  new_date = new_date.strftime("%Y-%m-%d")
  
  if stock_type == 'k':
    new_stock_data = ts.get_k_data(code, start=new_date)
  else:
    new_stock_data = ts.get_hist_data(code, start=new_date)

  new_stock_data.index = pd.to_datetime(new_stock_data['date'])
  new_stock_data = new_stock_data.sort_index(ascending=True)
  new_stock_data = new_stock_data[x_cols]
  
  stock_data = pd.concat((stock_data,new_stock_data))

  return stock_data

def download_stock_data(code,stock_type,start='1990'):
  if stock_type == 'k':
    stock_data = ts.get_k_data(code, start=start)
  else:
    stock_data = ts.get_hist_data(code, start=start)

  #x_cols = ['open','close','high','low','volume']
  #stock_data.index = pd.to_datetime(stock_data['date'])
  #stock_data = stock_data.sort_index(ascending=True)
  #stock_data = stock_data[x_cols]

  return stock_data

stock_basics_path = 'data/stock_basics.csv'
if not os.path.exists(stock_basics_path):
  print('downloading basic...')
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
  try:
    if os.path.exists(store_path):
      stock_data = update_stock_data(code, stock_type)
    else:
      stock_data = download_stock_data(code, stock_type)
  except:
    continue

  if stock_data is None:
    continue

  stock_data.to_csv(store_path)
  name = stock_basics.loc[code,'name']
  timeToMarket = stock_basics.loc[code,'timeToMarket']
  print("%d: %s(%s)\t-\t'%s'" % (i, name, code, timeToMarket))

  #if timeToMarket is not pd.to_datetime(np.nan):
  #  print(timeToMarket)
  #else:
  #  print('NaT')
  i += 1


