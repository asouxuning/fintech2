import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Model
import pandas as pd
import tensorflow as tf
import numpy as np

import stock_data as sd

batch_size = 100
path = 'data/hist/600000.csv'
stock_data = pd.read_csv(path)

train_x,test_x,train_y,test_y = sd.get_stock_samples(stock_data)

# 构建神经网络层 1层LSTM层+3层Dense层
# 用于1个输入情况
lstm_input = Input(shape=(30,6), name='lstm_input')
lstm_output = LSTM(128, activation=tf.atan, dropout_W=0.2, dropout_U=0.1)(lstm_input)
Dense_output_1 = Dense(64, activation='linear')(lstm_output)
Dense_output_2 = Dense(16, activation='linear')(Dense_output_1)
predictions = Dense(1, activation=tf.atan)(Dense_output_2)

model = Model(input=lstm_input, output=predictions)

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    
model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=10, verbose=2)

# 预测
predictions = model.predict(test_x)

# 预测值和真实值的关系
data1 = test_y
data2 = predictions
#fig, ax = plt.subplots(figsize=(8, 6))
#ax.plot(data2,data1, 'o', label="data")
#ax.legend(loc='best')
plt.plot(data2,data1,'o',label='data')
plt.legend(loc='best')
plt.show()

x = np.arange(len(test_y))
plt.plot(x,test_y,label='true data')
plt.plot(x,predictions,label='predict')
plt.legend()
plt.show()

accuracy = np.mean(((data2 > 0.0) == (data1 > 0.0)).astype(np.float32))
print(accuracy)
