import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import scale, minmax_scale 
from sklearn.decomposition import PCA

import stat_utils

def shufflelists(lists):
  # reindex
  ri = np.random.permutation(len(lists[0]))
  out = []
  for l in lists:
    out.append(l[ri])
  return out

def stock_data_add_return(stock_data, n_hold_days=1):
  # 将第n_days天的数据移到当前(第0天)
  # 将第1天的数据移到当前(第0天)
  # 并将它们进行运算
  # pandas的shift函数参數为正时,意味着将当下的数据向未来移(推)
  # pandas的shift函数参数为负时,意味着将未来的数据往当下移(拉)
  
  stock_data.loc[:,'return'] = stock_data['close'].shift(-n_hold_days) / stock_data['open'].shift(-1) - 1.0
  # 在当前的数据框中将缺失数据NaN都删除
  stock_data.dropna(inplace=True)

  return stock_data

x_cols=['open', 'close', 'high', 'low', 'volume'] 
y_cols=['close']

def set_stock_data_datetime_index(stock_data):
  stock_data.index = pd.to_datetime(stock_data['date']) 
  stock_data = stock_data.sort_index(ascending=True)
  return stock_data
  
def split_data_cols(stock_data, x_cols=x_cols, y_cols=y_cols):
  x = stock_data[x_cols]
  y = stock_data[y_cols]

  return x.values,y.values

def split_data_rows(stock_data, split_rate):
  split = int(len(stock_data)*split_rate)
  return (stock_data[:split], stock_data[split:])

def get_seq_samples(x,y,seq_len):
  x_list = []
  y_list = []
  for i in range(len(x)-(seq_len-1)):
    _x = x[i:i+seq_len]
    x_list.append(_x)

    _y = y[i+seq_len-1]
    y_list.append(_y)

  # 预计会有因样本不足导致的对空列表
  # 进行stack产生的ValueError
  try:
    x = np.stack(x_list)
    y = np.stack(y_list)
  except ValueError:
    raise

  return x,y
  
def scale_elem(X, scale=scale):
  X_list = []
  for x in X:
    X_list.append(scale(x))
  if len(X_list) > 0:
    return np.stack(X_list)
  else:
    return np.array([])
    
def get_stock_samples(stock_data, x_cols=x_cols, y_cols=y_cols, seq_len = 30): #, n_hold_days=5): 
  # 先升序排列,再添加收益率 
  #stock_data = set_stock_data_datetime_index(stock_data)
  
  #stock_data = stock_data_add_return(stock_data, n_hold_days)
  
  x,y = split_data_cols(stock_data, x_cols=x_cols, y_cols=y_cols)
  X,Y = get_seq_samples(x,y,seq_len)
  
  return X,Y

# 二值化股票的涨跌幅数据
def binarize_labels(Y,range_={-1.0,1.0}):
  Y = np.where(Y>0.0,max(range_),min(range_))
  return Y

# 将特征数据的每个样本进行拉直操作(ravel)
def ravel_feats(X):
  return X.reshape((X.shape[0],-1))

def get_list_stock_data(codes, stock_type):
  base = 'data/'+stock_type+'/'

  stock_data_list = []
  for code in codes:
    path = base+code+'.csv'
    stock_data = pd.read_csv(path)
    stock_data_list.append(stock_data)

  return stock_data_list
    
def get_list_stock_samples(stock_data_list):
  X_list = []
  Y_list = []
  for stock_data in stock_data_list:
    try:
      (X,Y) = get_stock_samples(stock_data)
      X_list.append(X)
      Y_list.append(Y)
    except:
      continue
  X = np.vstack(X_list)
  Y = np.vstack(Y_list)
  return (X,Y)

# c_name: concept name
def get_concepts_stock_samples(c_name):
  concepts  = pd.read_csv('data/concept_classified.csv', 
                          dtype={'code': str})
  concepts = concepts[concepts['c_name']==c_name]
  concepts = concepts['code']
  stock_data_list = get_list_stock_data(concepts, 'k')
  (X,Y) = get_list_stock_samples(stock_data_list)
  #X.dump('data/samples/X.npy')
  #Y.dump('data/samples/Y.npy')
  return (X,Y)
  
# 从股票数据中取得最新的n个样本
def get_latest_stock_features(stock_data, x_cols=x_cols, 
                              seq_len=30, n=1):
  features = []

  # 添加第一个样本
  stock_data = stock_data[x_cols]
  feature = stock_data[-seq_len:None]
  features.append(feature)
  for i in range(1,n):
    #截取倒数第i+1个元素到倒数第i+seq_len个元素的切片
    feature = stock_data[-(i+seq_len):-i]
    features.append(feature)

  return np.stack(features)
  
if __name__ == '__main__':
  stock_data = pd.read_csv('data/k/600000.csv')
  stock_data = set_stock_data_datetime_index(stock_data)
  features = get_latest_stock_features(stock_data, n=1)
  x_cols=['open', 'close', 'high', 'low', 'volume', 'return'] 
  (x,y) = get_stock_samples(stock_data, x_cols=x_cols)
  
  #df = pd.read_csv('data/k/600000.csv')
  #df = stock_data_add_return(df,n_hold_days=5)
  #x_cols=['open', 'close', 'high', 'low', 'volume'] 
  #df1 = df[x_cols]
  #(eigvals1,eigvecs1,weights1) = stat_utils.pca(df1.values)

  #x_cols=['open', 'close', 'high', 'low', 'volume', 'return'] 
  #df2 = df[x_cols]
  #(eigvals2,eigvecs2,weights2) = stat_utils.pca(df2.values)
  #
  #(X,Y) = get_concepts_stock_samples('特斯拉')
  #concepts  = pd.read_csv('data/concept_classified.csv', 
  #                        dtype={'code': str})
  #concepts = concepts[concepts['c_name']=='特斯拉']
  #concepts = concepts['code']
  #stock_data_list = get_list_stock_data(concepts, 'k')
  #(X,Y) = get_list_stock_samples(stock_data_list)
  #X.dump('data/samples/X.npy')
  #Y.dump('data/samples/Y.npy')

  ##path = 'data/k/600000.csv'
  #path = 'data/hist/600000.csv'
  #stock_data = pd.read_csv(path)

  ## 未来持有n_hold_days的天数 , 所以n_hold_days >= 1
  ## 当n_hold_days=1时,就是明天的收盘价close减去明天的开盘价
  #n_hold_days = 5
  #stock_data = stock_data_add_return(stock_data, n_hold_days)
  #
  #seq_len = 30
  #(X_train, Y_train, X_test, Y_test) = clean_stock_data(stock_data, 0.9, seq_len=seq_len, y_cols=['return'])
  #print(stock_data.head())
  #
  #Y_train = binarize_labels(Y_train)
  #Y_test = binarize_labels(Y_test)
