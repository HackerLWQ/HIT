#coding=utf-8
"""
演示内容：不同多项式回归的性能比较、使用岭回归克服过拟合现象
将500个点的前300个作为训练集，后200个作为测试集。
设置两个变量ridge（是否使用岭回归，为1表示使用）和sign（是否使用测试集测试，为1表示是）。
1.不使用岭回归（即不用正则处理），并在训练集上测试： ridge=0，sign=0
2.不使用岭回归（即不用正则处理），并在测试集上测试： ridge=0，sign=1
3.使用岭回归，在训练集上测试：ridge=1，sign=0
3.使用岭回归，在测试集上测试：ridge=1，sign=1

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#为0表示未使用岭回归，为1表示使用
ridge=1
#为0表示，在训练集上测试，改为1表示在测试集上测试    
sign=1
    
#arange函数用于创建等差数组,0为初始值,1为终值,0.002为步长。因此x为500个数值，与y共同形成500个点
x = np.arange(0, 1, 0.002)
#生成n个随机数可用rv_continuous.rvs(size=n)或rv_discrete.rvs(size=n)，
#其中rv_continuous表示连续型的随机分布，如均匀分布（uniform）、正态分布（norm）、贝塔分布（beta）等；
#rv_discrete表示离散型的随机分布，如伯努利分布（bernoulli）、几何分布（geom）、泊松分布（poisson）等。
#参数loc表示平均数，scale表示标准差，size是样本量
y = norm.rvs(loc=0, size=500, scale=0.1)
y = y + x**2

'''
#演示使用 np.newaxis 为 numpy.ndarray(多维数组)增加一个轴
a = np.arange(0, 1, 0.2)
print a
print a[np.newaxis]
print a[:, np.newaxis]

'''

def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

plt.scatter(x, y,marker='o',c='',edgecolors='g', s=30)
degree = [1,2,10]
y_test = []
y_test = np.array(y_test)

#分别试验1、2、100次多项式
for d in degree:
    #Pipeline可以将许多算法模型串联起来。 
    #参数格式：List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an estimator.
    clf = Pipeline([('poly', PolynomialFeatures(degree=d)),
                    ('linear',      LinearRegression(fit_intercept=False))])
    
    #训练集是前300个样本，测试集是后200个样本
    x_train=x[:300, np.newaxis]
    y_train=y[:300]
    
    x_test=x[301:, np.newaxis]
    y_test=y[301:]   


    if (ridge==0):
        clf.fit(x_train, y_train)
    else:
        clf = Pipeline([('poly', PolynomialFeatures(degree=d)),('linear', linear_model.Ridge ())])
        clf.fit(x_train, y_train)     


    print (x.shape)
    print (x[:, np.newaxis].shape)


    #考察在训练集上的表现（即将训练集作为测试集）
    if (sign==0) :
        y_predict = clf.predict(x_train) #先将训练集作为
        y_true= y_train

    #考察在真正测试集上的表现        
    else:
        y_predict = clf.predict(x_test)
        y_true= y_test

    
    print('when degree is %d: rmse=%.2f, R2=%.2f' %
      (d,
       rmse(y_predict, y_true),
       R2(y_predict, y_true),
       ))    
    
    print(clf.named_steps['linear'].coef_)
    if (sign==0):

        if (d==1):
            plt.plot(x_train, y_predict, linewidth=3,linestyle='dashed',color='black')
        if (d==2):
            plt.plot(x_train, y_predict, linewidth=3,linestyle='dashed',color='r')
        if (d==10):
            plt.plot(x_train, y_predict, linewidth=3,linestyle='dashed',color='b')
        
    else:
        if (d==1):
            plt.plot(x_test, y_predict, linewidth=3,linestyle='dashed',color='black')
        if (d==2):
            plt.plot(x_test, y_predict, linewidth=3,linestyle='dashed',color='r')
        if (d==10):
            plt.plot(x_test, y_predict, linewidth=3,linestyle='dashed',color='b')
    
plt.grid()
plt.legend(['1','2','10'], loc='upper left')
plt.show()

