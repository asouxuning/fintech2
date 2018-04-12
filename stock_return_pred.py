import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sys
import stock_data as sd

dtype = tf.float32

path = 'data/hist/600000.csv'
stock_data = pd.read_csv(path)

X_train,X_test,Y_train,Y_test = sd.get_stock_samples(stock_data)

n_input = X_train.shape[-1]
seq_len = X_train.shape[-2]
n_output = Y_train.shape[-1]
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

Y_pred = tf.identity(tf.matmul(Z,W)+b)

# loss
loss = tf.reduce_mean(tf.square(Y_pred-Y))

# 
accuracy = tf.reduce_mean(tf.cast(
                           tf.equal((tf.greater(Y_pred,0.0)),
                                    (tf.greater(Y,0.0))
                              ),
                           tf.float32))
# optimizer
lr = 0.01
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  losses = []
  accs = []
  epochs = 1000
  batch_size = 64 #X_train.shape[0]
  vali_feed = {X: X_test, Y: Y_test}
  train_feed = {X: X_train, Y: Y_train}
  for i in range(epochs):
    X_train,Y_train = sd.shufflelists([X_train,Y_train])
    batchs = X_train.shape[0] // batch_size + 1

    #train_loss = 0.0
    for k in range(batchs):
      X_batch = X_train[k*batch_size:(k+1)*batch_size]
      Y_batch = Y_train[k*batch_size:(k+1)*batch_size]
      batch_feed = {X: X_batch, Y: Y_batch}

      # 不能喂空训练集,否则产生NaN
      if X_batch.shape[0] == 0:
        continue

      sess.run(train, feed_dict=batch_feed)
      #train_loss += sess.run(loss, feed_dict=batch_feed)

    vali_loss = sess.run(loss, feed_dict=vali_feed)
    train_loss = sess.run(loss, feed_dict=train_feed)

    #train_loss /= batchs
    losses.append([train_loss,vali_loss])

    train_acc = sess.run(accuracy, feed_dict=train_feed)
    vali_acc = sess.run(accuracy, feed_dict=vali_feed)
    accs.append([train_acc,vali_acc])

    print("%d: %f - %f, %f - %f" % (i, train_loss, vali_loss, train_acc, vali_acc))
    #print("%d: %f - %f" % (i, train_loss, test_loss))

  losses = np.vstack(losses)
  accs = np.vstack(accs)

  plt.plot(losses[:,0], label='train')
  plt.plot(losses[:,1], label='test')
  plt.legend()
  plt.show()

  plt.plot(accs[:,0], label='train')
  plt.plot(accs[:,1], label='test')
  plt.legend()
  plt.show()

  train_pred = sess.run(Y_pred, feed_dict={X: X_train})
  plt.plot(Y_train, label='true')
  plt.plot(train_pred, label='predict')
  plt.legend()
  plt.show()

  test_pred = sess.run(Y_pred, feed_dict={X: X_test})
  plt.plot(Y_test, label='true')
  plt.plot(test_pred, label='predict')
  plt.legend()
  plt.show()

  X_data = np.vstack((X_train,X_test))
  Z_ = sess.run(Z, feed_dict={X: X_data})

Y_data = np.vstack((Y_train,Y_test))
labels = np.where(Y_data>0.0, 1.0, -1.0)
## sklearn.svm.SVC的label需要一维张量
labels = labels.reshape((labels.shape[0],))

# svm classification
z0 = Z_[:,0]
z1 = Z_[:,1]
l = len(z0)
z0 = z0.reshape((l,))
z1 = z1.reshape((l,))

colors = np.where(labels==1.0, 'red', 'green')
plt.scatter(z0, z1, color=colors)
plt.show()

train_correct = np.mean(((train_pred>0.0) == (Y_train>0.0)).astype(np.float32))
print(train_correct)

test_correct = np.mean(((test_pred>0.0) == (Y_test>0.0)).astype(np.float32))
print(test_correct)

## 样本分配比例
#Y_train = Y_train.reshape((Y_train.shape[0],))
#Y_test = Y_test.reshape((Y_test.shape[0],))
#print(len(Y_train[Y_train>0.0])/len(Y_train)) 
#print(len(Y_test[Y_test>0.0])/len(Y_test)) 
