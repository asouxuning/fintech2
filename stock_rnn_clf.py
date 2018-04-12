import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stock_data as sd

import sys

#path = 'data/k/600000.csv'
path = 'data/hist/600000.csv'
stock_data = pd.read_csv(path)

# 未来持有n_hold_days的天数 , 所以n_hold_days >= 1
# 当n_hold_days=1时,就是明天的收盘价close减去明天的开盘价
n_hold_days = 1

X_train,X_test,Y_train,Y_test = sd.get_stock_samples(stock_data, n_hold_days=n_hold_days)

# 对标签进行0,1二值化处理
range_ = {0.0,1.0}
Y_train = sd.binarize_labels(Y_train, range_=range_)
Y_test = sd.binarize_labels(Y_test, range_=range_)

dtype = tf.float32
n_input = X_train.shape[-1]
n_output = Y_train.shape[-1]
seq_len = X_train.shape[-2]
n_state = 2

# 
X = tf.placeholder(dtype, (None, seq_len, n_input))
Y = tf.placeholder(dtype, (None, n_output))

cell = tf.nn.rnn_cell.LSTMCell(num_units=n_state, 
                               activation=tf.nn.sigmoid, 
                               initializer=tf.orthogonal_initializer(), 
                               name="rnn_cell")
outputs,states = tf.nn.dynamic_rnn(cell, X, dtype=dtype)

Z = outputs[:,-1]
W = tf.get_variable('W', shape=(n_state,n_output), 
                    dtype=dtype, 
                    initializer=tf.glorot_uniform_initializer())
b = tf.Variable(tf.zeros((n_output,), dtype=dtype))

Y_pred = tf.nn.sigmoid(tf.matmul(Z,W)+b)

confidence = 0.5
accuracy = tf.reduce_mean(tf.cast(
                 tf.equal(Y_pred>confidence,Y>confidence), tf.float32))
loss = tf.reduce_mean(-Y*tf.log(Y_pred)-(1-Y)*tf.log(1-Y_pred))

# optimizer
lr = 0.01
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)

  epochs = 1001
  batch_size = 128
  loss_list = []
  acc_list = []
  for i in range(epochs):
    X_train,Y_train = sd.shufflelists([X_train,Y_train])
    batchs = len(X_train) // batch_size + 1

    train_loss = 0.0
    for k in range(batchs):
      X_ = X_train[k*batch_size:(k+1)*batch_size]
      Y_ = Y_train[k*batch_size:(k+1)*batch_size]
      if X.shape[0] == 0:
        continue

      feed_dict={X:X_,Y:Y_}
      _ = sess.run(train, feed_dict=feed_dict) 
      train_loss += sess.run(loss, feed_dict=feed_dict) 
    
    train_loss /= batchs
    test_loss = sess.run(loss, feed_dict={X: X_test, Y: Y_test}) 
    loss_list.append([train_loss,test_loss])

    acc_train = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
    acc_test = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
    acc_list.append([acc_train,acc_test])

    print("%d: %f - %f - %f - %f" % (i, train_loss, test_loss, acc_train, acc_test))
   
  losses = np.stack(loss_list)
  accs = np.stack(acc_list)
  x = np.arange(losses.shape[0])
  pred = sess.run(Y_pred, feed_dict={X: X_test, Y: Y_test})
  
  plt.plot(x,losses[:,0],label='train loss')
  plt.plot(x,losses[:,1],label='test loss')
  plt.legend()
  plt.show()

  plt.plot(x,accs[:,0],label='train accuracy')
  plt.plot(x,accs[:,1],label='test accuracy')
  plt.legend()
  plt.show()
