import theano
import theano.tensor as T
import numpy as np

from sklearn.model_selection import train_test_split

def glorot_init(n_in, n_out, dtype=theano.config.floatX):
  w = np.random.normal(loc=0, 
                       scale=2.0 / (n_in + n_out),
                       size=(n_in, n_out))
  w = w.astype(dtype)
  return w
  
def bias_init(dim, dtype=theano.config.floatX):
  b = np.zeros((dim,), dtype=dtype)
  return b

def rnn_step(x,s,W,U,bh,V,by):
  t = T.nnet.sigmoid(T.dot(s,W)+T.dot(x,U)+bh)
  o = T.tanh(T.dot(t,V)+by)
  return o,t
   
# 获得两个张量同对应元素同号的比率
def same_sign_rate(X,Y):
  equals = T.eq(T.gt(X,0.0),T.gt(Y,0.0)),
  accuracy = T.mean(T.cast(equals, 'int8'))
  return accuracy

class RNN(object):
  # 跟据样本数集来构建模型
  def __init__(self, 
               num_units, 
               step=rnn_step, 
               dtype=theano.config.floatX):

    self.n_h = num_units
    self.step = rnn_step

    self.dtype = dtype

  def inference(self):
    X = T.tensor3('X', dtype=self.dtype)

    # 将输入张量由(batch_size,seq_len,input_dim)
    # 变形为(seq_len,batch_size,input_dim)
    X_ = X.dimshuffle([1,0,2])
    
    # 隐藏层到状态层的全连接层
    W = theano.shared(glorot_init(self.n_h,self.n_h))
    # 输入层到隐藏层的全连接层
    U = theano.shared(glorot_init(self.n_in,self.n_h))
    # 隐藏层的偏移
    bh = theano.shared(bias_init(self.n_h))
  
    # 状态到输出的全连接层
    V = theano.shared(glorot_init(self.n_h,self.n_out))
    # 输出层的偏移
    by = theano.shared(bias_init(self.n_out))
    
    params = [W, U, bh, V, by]
    self.params = params
  
    # 初始隐藏层为(n_h,)的全0向量
    h0 = theano.shared(np.zeros((self.n_h,),dtype=self.dtype))
    # 当用mini-batch来匹量训练模型时,
    # 初始隐藏层的第一维(first dimemsion)可能会变化,
    # 由其是最后一个batch往往会有更小的样本量,
    # 所以RNN的初始状态要能根据实际输入数据的样本量,
    # 来决定其第一维的大小
    batch_size = X_.shape[1]
    H = T.alloc(h0, batch_size, self.n_h)
  
    (os,ts),_ = theano.scan(fn=self.step,
                            sequences=X_,
                            outputs_info=[None,H],
                            non_sequences=params,
                            strict=True)
    Y_ = os[-1]

    # 根据由前向传播推导出的结果张量来决定接收输入的
    # 标签张量Y的张量大小
    Y = T.TensorType(dtype=Y_.dtype, 
                     broadcastable=Y_.broadcastable)\
                     (name='Y')

    self.X = X
    self.Y_ = Y_
    self.Y = Y

    return Y_
  
  def cost(self):
    Y = self.Y
    Y_ = self.Y_
    error = T.mean((Y-Y_)**2)
    self.error = error
    return error
    
  def accuracy(self):
    acc = same_sign_rate(self.Y,self.Y_)
    self.acc = acc
    return acc
    
  def optimize(self, lr):
    error = self.error
    params = self.params

    grads = T.grad(cost=error, wrt=params)
    updates = [(w,w-lr*g) for (w,g) in zip(params,grads)]

    self.updates = updates
    return updates

  def compile(self):
    train_ = theano.function(inputs=[self.X,self.Y], 
                            outputs=self.error,
                            updates=self.updates)
    predict_ = theano.function(inputs=[self.X], 
                              outputs=self.Y_)
    #get_error_ = train_.copy(delete_updates=True)
    get_error_ = theano.function(inputs=[self.X,self.Y], 
                            outputs=self.error)
    get_accuracy_ = theano.function(inputs=[self.X,self.Y], 
                            outputs=self.acc)

    self.train_ = train_
    self.predict_ = predict_
    self.get_error_ = get_error_
    self.get_accuracy_ = get_accuracy_

  def train(self, x, y, batch_size, epochs):
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    train_size = x_train.shape[0]
    batchs = (train_size // batch_size) + (0 if train_size % batch_size == 0 else 1)

    for i in range(epochs):
      for k in range(batchs):
        x_batch = x_train[k*batch_size:(k+1)*batch_size]
        y_batch = y_train[k*batch_size:(k+1)*batch_size]
        loss = self.train_(x_batch,y_batch)

    acc = self.get_accuracy_(x_test,y_test)
    #self.get_error_(x_test,y_test)

    return acc 
    
  def fit(self, x, y, lr, batch_size=1, epochs=1):
    self.x = x
    self.y = y

    self.n_in= x.shape[-1]
    self.n_out= y.shape[-1]

    self.inference()
    self.cost()
    self.accuracy()
    self.optimize(lr)
    self.compile()

    return self.train(x,y,batch_size, epochs)
  
  def predict(self,x):
    return self.predict_(x)
