import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import scale, minmax_scale 
from sklearn.decomposition import PCA

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
  for i in range(len(x)-seq_len):
    _x = x[i:i+seq_len]
    x_list.append(_x)

    _y = y[i+seq_len]
    y_list.append(_y)

  x = np.stack(x_list)
  y = np.stack(y_list)
  return x,y
  
def scale_elem(X, scale=scale):
  X_list = []
  for x in X:
    X_list.append(scale(x))
  if len(X_list) > 0:
    return np.stack(X_list)
  else:
    return np.array([])
    
def get_stock_samples(stock_data, seq_len = 30, n_hold_days=5, split_rate=0.8):
  # 先升序排列,再添加收益率 
  stock_data = set_stock_data_datetime_index(stock_data)
  
  stock_data = stock_data_add_return(stock_data, n_hold_days)
  
  x_cols=['open', 'close', 'high', 'low', 'volume'] 
  x,y = split_data_cols(stock_data, x_cols=x_cols, y_cols=['return'])
  dataX,dataY = get_seq_samples(x,y,seq_len)
  dataX,dataY = shufflelists([dataX,dataY])
  
  (X_train,X_test) = split_data_rows(dataX, split_rate)
  (Y_train,Y_test) = split_data_rows(dataY, split_rate)

  # 将特征数据归一化
  X_train = scale_elem(X_train)
  X_test = scale_elem(X_test)

  return (X_train,X_test,Y_train,Y_test)

def clean_stock_data(stock_data, split_rate=0.7, seq_len=7,
                     x_cols=x_cols, y_cols=y_cols):

  stock_data = set_stock_data_datetime_index(stock_data)
  x,y = split_data_cols(stock_data, x_cols=x_cols, y_cols=y_cols)

  dataX,dataY = get_seq_samples(x,y,seq_len)

  (trainX,testX) = split_data_rows(dataX, split_rate)
  (trainY,testY) = split_data_rows(dataY, split_rate)

  trainX = scale_elem(trainX)
  testX = scale_elem(testX)
  
  return (trainX,trainY,testX,testY)

# 二值化股票的涨跌幅数据
def binarize_labels(Y,range_={-1.0,1.0}):
  Y = np.where(Y>0.0,max(range_),min(range_))
  return Y

# 将特征数据的每个样本进行拉直操作(ravel)
def ravel_feats(X):
  return X.reshape((X.shape[0],-1))

def get_list_stock_data(stock_type):
  path = 'data/'+stock_type+'/'
  files = os.listdir(path)
  X_list = []
  Y_list = []
  for f in files:
    file_path = path + f
    print(file_path)
    stock_data = pd.read_csv(file_path)
    try: 
      (X,_,Y,_) = get_stock_samples(stock_data,split_rate=1.0) 
    except:
      continue

    X_list.append(X)
    Y_list.append(Y)
  
  X = np.vstack(X_list)
  Y = np.vstack(Y_list)

  return X,Y

if __name__ == '__main__':
  X,Y = get_list_stock_data('k')
  X.dump('X.npy')
  Y.dump('Y.npy')

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
