import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import stock_data as sd

import sys

#path = 'data/k/600000.csv'
path = 'data/hist/600000.csv'
stock_data = pd.read_csv(path)

seq_len = 30
#(X_train, Y_train, X_test, Y_test) = sd.clean_stock_data(stock_data, 0.8, seq_len=seq_len, y_cols=['return'])

# 先升序排列,再添加收益率 
stock_data = sd.set_stock_data_datetime_index(stock_data)

# 未来持有n_hold_days的天数 , 所以n_hold_days >= 1
# 当n_hold_days=1时,就是明天的收盘价close减去明天的开盘价
n_hold_days = 5
stock_data = sd.stock_data_add_return(stock_data, n_hold_days)

x_cols=['open', 'close', 'high', 'low', 'volume', 'turnover'] 
x,y = sd.split_data_cols(stock_data,x_cols=x_cols,y_cols=['return'])

dataX,dataY = sd.get_seq_samples(x,y,seq_len)
dataX,dataY = sd.shufflelists([dataX,dataY])


split_rate = 0.8
(X_train,X_test) = sd.split_data_rows(dataX, split_rate)
(Y_train,Y_test) = sd.split_data_rows(dataY, split_rate)

# 将特征数据归一化
X_train = sd.scale_elem(X_train)
X_test = sd.scale_elem(X_test)

# 将每个特征样本整形为向量
# SVM要处理的是向量
X_train = sd.ravel_feats(X_train)
X_test = sd.ravel_feats(X_test)

# 对标签进行二值化处理
Y_train = sd.binarize_labels(Y_train)
Y_test = sd.binarize_labels(Y_test)

# sklearn.svm要求输入标签是1D数组
# 所以要将标签列向量压平(ravel)
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()

for degree in range(10):
  #degree = 3
  clf = SVC(C=1.0, kernel='poly', degree=degree)
  clf.fit(X_train,Y_train)
  train_score = clf.score(X_train,Y_train)
  test_score = clf.score(X_test,Y_test)
  print('degree:%d - %f - %f' % (degree, train_score, test_score))

