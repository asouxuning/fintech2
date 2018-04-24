import theano
import theano.tensor as T
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sys
import stock_data as sd
from rnn2 import RNN
import time

batch_size = 64

def get_stock_data(code, dtype=theano.config.floatX):
  path = 'data/k/'+code+'.csv'
  stock_data = pd.read_csv(path,
                           dtype={'date': str,
                                  'open': dtype,
                                  'close': dtype,
                                  'high': dtype,
                                  'low': dtype,
                                  'volume': dtype,
                                  'return': dtype
                           }
               )
  # 将时间作为索引,并按时间进行升序排列
  stock_data = sd.set_stock_data_datetime_index(stock_data)
  # 添加回报率return作为数据框的一列
  stock_data = sd.stock_data_add_return(stock_data, n_hold_days=5)

  return stock_data
  
def get_stock_samples(stock_data):
  x_cols=['open', 'close', 'high', 'low', 'volume', 'return'] 
  y_cols=['return'] 
  
  (x,y) = sd.get_stock_samples(stock_data, x_cols=x_cols,
                               y_cols=y_cols)
  x = sd.scale_elem(x)
  
  x_new = sd.get_latest_stock_features(stock_data, 
                                       x_cols=x_cols,
                                       n=5)
  x_new = sd.scale_elem(x_new)
  return x,y,x_new
  
def get_stock_data_set(code,with_vali=True):
  path = 'data/k/'+code+'.csv'
  stock_data = pd.read_csv(path)
  # 将时间作为索引,并按时间进行升序排列
  stock_data = sd.set_stock_data_datetime_index(stock_data)
  # 添加回报率return作为数据框的一列
  stock_data = sd.stock_data_add_return(stock_data, n_hold_days=5)
  
  x_cols=['open', 'close', 'high', 'low', 'volume', 'return'] 
  y_cols=['return'] 
  
  (x,y) = sd.get_stock_samples(stock_data, x_cols=x_cols,
                               y_cols=y_cols)
  x = sd.scale_elem(x)
  
  x_new = sd.get_latest_stock_features(stock_data, 
                                       x_cols=x_cols,
                                       n=5)
  x_new = sd.scale_elem(x_new)
  
  #x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=False)
  test_size = 100
  x_test = x[-test_size:None]
  y_test = y[-test_size:None]
  x_train = x[:-test_size]
  y_train = y[:-test_size]

  if with_vali == True:
    # 训练集中再分出验证集
    x_train,x_vali,y_train,y_vali = train_test_split(x_train,y_train)
  else:
    x_vali = None
    y_vali = None
  
  data_set = {'orig': {'x': x, 'y': y},
              'train': {'x': x_train, 'y': y_train},
              'test': {'x': x_test, 'y': y_test},
              'vali': {'x': x_vali, 'y': y_vali}
             }
              
  return data_set,x_new
    
if __name__ == '__main__':
  argv = sys.argv
  if len(argv) < 2:
    print('please enter the cmd')
    sys.exit()

  cmd = argv[1]

  if cmd == 'code':
    #code = '600000'
    arg = argv[2]
    code = arg.strip()
    basics = pd.read_csv('data/stock_basics.csv',
                         dtype={'code': str})
    basics = basics[basics['code'] == code]
    stocks = basics[['code','name']]
  elif cmd == 'concept':
    #c_name = '特斯拉'
    arg = argv[2]
    c_name = arg.strip()
    concepts  = pd.read_csv('data/concept_classified.csv', 
                            dtype={'code': str})
    concepts = concepts[concepts['c_name']==c_name]
    stocks = concepts[['code','name']]
  elif cmd == 'industry':
    #c_name = '金融行业'
    arg = argv[2]
    c_name = arg.strip()
    industry = pd.read_csv('data/industry_classified.csv',
                           dtype={'code': str})
    industry = industry[industry['c_name']==c_name]
    stocks = industry[['code','name']]
  elif cmd == 'all':
    basics = pd.read_csv('data/stock_basics.csv',
                         dtype={'code': str})
    stocks = basics[['code','name']]
  else:
    print('not a correct command')
    sys.exit()
    
  stocks = zip(stocks['code'],stocks['name'])

  #companies = {}
  companies =[]
  date = time.strftime("%Y-%m-%d", 
                       time.localtime(time.time()))
  i = 0
  for (code,name) in stocks:
    try:
      dtype = theano.config.floatX
      stock_data = get_stock_data(code, dtype=dtype)
      x,y,_ = get_stock_samples(stock_data)
      
      model = RNN(2)
      acc = model.fit(x,y,0.5, 
                      batch_size=64, 
                      epochs=100)
    
      rets = model.predict(_)
      print("%d: %s(%s):\t%.4f(%.1f%%)" % (i, name, code, round(rets[-1][0],4), round(acc*100,1)))

      company = (code, name, float(rets[-1]), float(acc))
      companies.append(company)

      i += 1
    except ValueError as e:
      continue
    
  companies_sorted = sorted(companies,
                            key=lambda item: item[2], 
                            reverse=True)
  df = pd.DataFrame(companies_sorted)
  df.columns = ['code', 'name', 'return', 'accuracy']
  df.to_csv('report/report_'+date+'.csv')
