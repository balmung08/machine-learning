import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

#反向传播神经网络
#读取数据集
data = loadmat('usedata/ex4data1.mat')
x,y = data['X'], data['y']
weight = loadmat("usedata/ex4weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播函数
def forward_propagate(X, theta1, theta2):
    X,theta1,theta2 = np.matrix(X),np.matrix(theta1),np.matrix(theta2)
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)#在X插入一列1用于与theta0计算 a1(5000, 401)
    z2 = a1 * theta1.T #线性计算结果 theta1(25, 401) z2(5000, 25)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)#在激活函数计算结果后再插一列1 a2(5000, 26) theta2(10, 26)
    z3 = a2 * theta2.T #第二层线性计算 z3(5000, 10)
    h = sigmoid(z3) #激活后给到第三层
    return a1, z2, a2, z3, h

#代价函数
def cost(theta1, theta2, X, y):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m
    return J

#正则化代价函数
def costReg(theta1, theta2, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J


#sigmoid梯度，即激活函数的导数
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


#反向传播函数
def backprop(params, input_size, hidden_size, num_labels, X, y):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    theta1, theta2 = weight['Theta1'], weight['Theta2']
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m
    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)
        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    return J, delta1, delta2

#正则化反向传播函数
def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)
        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J, grad

#计算部分
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
#随机初始
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print(cost(theta1, theta2, x, y_onehot))
print(costReg(theta1, theta2, x, y_onehot,learning_rate))
print(backprop(params, input_size, hidden_size, num_labels, x, y))
print(backpropReg(params, input_size, hidden_size, num_labels, x, y,learning_rate))

'''
#使用scipy计算并评估
from scipy.optimize import minimize
fmin = minimize(fun=backpropReg, x0=(params), args=(input_size, hidden_size, num_labels, x, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)
x = np.matrix(x)
thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(x, thetafinal1, thetafinal2 )
y_pred = np.array(np.argmax(h, axis=1) + 1)
#评价报告
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''

#隐藏层可视化
#(个人感觉没有太大意义)