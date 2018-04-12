import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import binarize, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import stock_data as sd
import sys

path = 'data/hist/600000.csv'
stock_data = pd.read_csv(path)

X,Y = sd.get_stock_samples(stock_data,x_cols=['close'])

X = sd.scale_elem(X)
X = X.reshape((X.shape[0],-1))

Y = binarize(Y)
Y = Y.reshape((Y.shape[0],))

x_train,x_test,y_train,y_test = train_test_split(X,Y)


n_input = x_train.shape[-1]
seq_len = x_train.shape[-2]
n_output = y_train.shape[-1]
n_state = 2

for i in range(10):
  clf = SVC(C=1.0, kernel='poly', degree=i)
  clf.fit(x_train,y_train)
  train_score = clf.score(x_train,y_train)
  test_score = clf.score(x_test,y_test)

  print('%d: %f - %f' % (i, train_score, test_score))
  
print()
for i in range(10):
  clf = SVC(C=1.0, kernel='poly', degree=i)
  scores = cross_val_score(clf, X, Y, cv=5)
  print("%f" % scores.mean())

print()
for i in range(10):
  clf = SVC(C=1.0, kernel='poly', degree=i)
  strKFold = StratifiedKFold(n_splits=5, shuffle=False)
  scores = cross_val_score(clf,X,Y,cv=strKFold)
  print("%f" % scores.mean())
  
