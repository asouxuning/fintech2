import stock_data as sd
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import *

path = 'data/k/600000.csv'
stock_data = pd.read_csv(path)

stock_data = sd.set_stock_data_datetime_index(stock_data)
stock_data = sd.stock_data_add_return(stock_data,n_hold_days=5)

ret = stock_data['return']

plt.plot(ret.index, ret)
plt.show()

lags = 30
plot_acf(ret, lags=lags)
plt.show()

plot_pacf(ret, lags=lags)
plt.show()
