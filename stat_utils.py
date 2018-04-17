import numpy as np
import pandas as pd
import stock_data
from sklearn.preprocessing import scale

def pca(X):
  X = scale(X)
  cov = np.cov(X.T)
  eigvals,eigvecs = np.linalg.eig(cov)
  weights = eigvals / sum(eigvals) 
  return (eigvals,eigvecs,weights)

#if __name__ == '__main__':
#  cols=['close', 'open', 'high', 'low', 'volume'] 
#  stock_data = pd.read_csv('data/hist/000009.csv')
#  stock_data = stock_data[cols]
#  eigvals,eigvecs,weights = pca(stock_data)
  
