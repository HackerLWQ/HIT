#coding:utf-8
# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
import os


#%matplotlib inline  使用 rc_params 函数，它返回一个配置字典：
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'# 差值方式
plt.rcParams['image.cmap'] = 'gray'#灰度图

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

np.random.seed(0)
N = 100 # number of points per class 每个类中的点的个数
D = 2 # dimensionality 维度，2维.每个数据点是二维的
K = 3 # number of classes 类的个数，3类
##先临时生成一个300行2列（点数，维度）的全0矩阵，用于存放数据点。这些数据值马上会更新。
#注意，参数是一个tuple，所以有两个括号。完整的形式为：zeros(shape,dtype=)。
X = np.zeros((N*K,D)) 

# class labels 类的标签#创建指定长度(300)的全0数组 ，8位无符号整数 
y = np.zeros(N*K, dtype='uint8') 

for j in range(K):
  ix = range(N*j,N*(j+1))#  输出0-99， 100-199，200-299三个段内的整数
  r = np.linspace(0.0, 1, N) # radius 在从0到1中均匀地产生100个数：
  
  # theta 在 0-4， 4-8， 8-12，每个范围之间均匀地产生100个数,再加上随机数0.2 randn：  
  #filled with random floats sampled from a univariate “normal” (Gaussian) distribution of 
  #mean 0 and variance 1 
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] #x是个数组，分别表示横、纵坐标
  y[ix] = j

fig = plt.figure()
#c='y' 试一下，则颜色不可区分 c表示颜色序列， s表示点的大小 cmap里面的参数为一种布局方式
#y在此处既代表点的种类，也代表点的颜色，
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral) #X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据，第二维中取第0个数据，直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
plt.xlim([-1,1]) #返回当前的X轴绘图范围。
plt.ylim([-1,1])
plt.legend()
plt.show()
fig.savefig('spiral_raw.png')

#Train a Linear Classifier

# initialize parameters randomly  #randn函数生成一些正态分布的随机数组（D行K列）
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))
#print "np.dot(X, W)"
#print np.dot(X, W)
print ("b is...")
print (b)

  
# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0] #行数 300行
#迭代次数
for i in range(200):
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b   
  
  # compute the class probabilities。 exp:返回e的n次方,e是一个常数为2.71828
  exp_scores = np.exp(scores) #scores为300*3的矩阵
  #exp_scores=scores
  print ("scores is:")
  print (scores.shape)

  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  print ("prob is...")
  print (probs)
 
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  print ("corect_logprobs is..")
  print (corect_logprobs  )
  #os.system("pause")   
  data_loss = np.sum(corect_logprobs)/num_examples
  print ("W*W is...")
  print (W)
  print (W*W)
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print ("iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  
  # evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

#meshgrid是用于生成网格采样点的函数。
#在计算机中进行绘图操作时，往往需要一些采样点，然后根据这些采样点来绘制出整个图形。
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
                     
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)


fig = plt.figure()
# contourf用于画三维等高线图,并会对等高线间的区域进行填充
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
fig.savefig('spiral_linear.png')



#现在改造为神经网络
# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print ("iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
  dW = np.dot(X.T, dhidden)
  #按列相加，并且保持其二维特性。keepdims主要用于保持矩阵的二维特性
  db = np.sum(dhidden, axis=0, keepdims=True)
  
  # add regularization gradient contribution
  dW2 += reg * W2
  dW += reg * W
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2
  
  # evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
#fig.savefig('spiral_net.png')