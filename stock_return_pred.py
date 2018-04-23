import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sys
import stock_data as sd

dtype = tf.float32
batch_size = 64

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
              
  print('sample size: ', x.shape[0])

  return data_set,x_new

def get_stock_data_info(data_set):
  x_train = data_set['train']['x']
  y_train = data_set['train']['y']

  # 通过数据集获得训练张量的参数
  n_input = x_train.shape[-1]
  seq_len = x_train.shape[-2]
  n_output = y_train.shape[-1]

  return n_input, n_output, seq_len 


def train_and_predict(code,data_set,x_new,verbose=True,show_graph=False):
  n_state = 3
  batch_size = 64

  x = data_set['orig']['x']
  y = data_set['orig']['y']

  x_train = data_set['train']['x']
  y_train = data_set['train']['y']
  x_test = data_set['test']['x']
  y_test = data_set['test']['y']
  x_vali = data_set['vali']['x']
  y_vali = data_set['vali']['y']

  # 训练集大小
  train_len = x_train.shape[0]
  # 一轮epoch有batchs次训练
  batchs = (train_len // batch_size) + (0 if train_len % batch_size == 0 else 1)

  n_input, n_output, seq_len = get_stock_data_info(data_set)
  
  # 
  X = tf.placeholder(dtype, (None, seq_len, n_input))
  Y = tf.placeholder(dtype, (None, n_output))

  with tf.variable_scope(code):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=n_state, 
                    activation=tf.nn.sigmoid, 
                    initializer=tf.orthogonal_initializer()) 
    #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_state)
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
                ), tf.float32))
    
    # optimizer
    learning_rate = 0.5 
    decay_rate = 0.99
    global_step = tf.Variable(tf.constant(0), trainable=False)

    lr = tf.train.exponential_decay(learning_rate=learning_rate,
                                    global_step=global_step,
                                    # 每一个epoch更新一次学习率
                                    decay_steps=batchs,
                                    decay_rate=decay_rate)
  
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()

  config=tf.ConfigProto(log_device_placement=True)
  with tf.Session(config=None) as sess:
    sess.run(init)

    losses = []
    accs = []
    epochs = 100
    if x_vali is not None:
      vali_feed = {X: x_vali, Y: y_vali}
    train_feed = {X: x_train, Y: y_train}
    for i in range(epochs):
      for k in range(batchs):
        x_batch = x_train[k*batch_size:(k+1)*batch_size]
        y_batch = y_train[k*batch_size:(k+1)*batch_size]
        batch_feed = {X: x_batch, Y: y_batch}
  
        sess.run(train, feed_dict=batch_feed)
  
      train_loss = sess.run(loss, feed_dict=train_feed)
      if x_vali is not None:
        vali_loss = sess.run(loss, feed_dict=vali_feed)
        losses.append([train_loss,vali_loss])
      else:
        losses.append([train_loss])
  
      train_acc = sess.run(accuracy, feed_dict=train_feed)
      if x_vali is not None:
        vali_acc = sess.run(accuracy, feed_dict=vali_feed)
        accs.append([train_acc,vali_acc])
      else:
        accs.append([train_acc])
  
      if verbose == True:
        if x_vali is not None:
          print("%d: %f - %f, %f - %f" % (i, train_loss, vali_loss, train_acc, vali_acc))
        else:
          print("%d: %f - %f" % (i, train_loss, train_acc))
      #print("%d: %f" % sess.run((global_step,lr)))
  
    losses = np.vstack(losses)
    accs = np.vstack(accs)
  
    test_feed = {X: x_test, Y: y_test}
    test_loss = sess.run(loss, feed_dict=test_feed)
  
    # 把在测试集上得到的正确率作为可信度
    confidence = sess.run(accuracy, feed_dict=test_feed)
  
    if show_graph == True:
      n_period = 30
      x_pred = sess.run(Y_pred, feed_dict={X: x})
      plt.plot(y[-n_period:None], label='true')
      plt.plot(x_pred[-n_period:None], label='predict')
      plt.legend()
      plt.show()

    rets = sess.run(Y_pred, feed_dict={X: x_new})
    rets = rets.reshape((rets.shape[0],))
    rets = np.flip(rets,axis=0)

  return confidence, rets

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

  res = []
  for (code,name) in stocks:
    try:
      data_set,x_new = get_stock_data_set(code,with_vali=False)
    except:
      continue
    confidence,rets = train_and_predict(code,data_set,
                                  x_new,verbose=False,
                                  show_graph=True)
    res.append((code,confidence,rets))

    print("%s(%s):\t%f(%f)" % (name, code, rets[-1], confidence))
    #plt.plot(rets)
    #plt.text(0, 0, confidence, fontsize=24)
    #plt.show()

  res = sorted(res, key=lambda r: r[2][-1], reverse=True)

