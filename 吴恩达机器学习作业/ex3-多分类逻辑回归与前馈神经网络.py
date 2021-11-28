#1.多分类逻辑回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as scio

data = scio.loadmat("usedata/ex3data1.mat")
pic = data["X"]
label = data["y"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 实现正则化的代价函数
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

#正则化逻辑回归+向量化
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()

#scipy一对多分类器
from scipy.optimize import minimize
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    all_theta = np.zeros((num_labels, params + 1))
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x
    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    h = sigmoid(X * all_theta.T)
    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1
    return h_argmax

#主执行部分
all_theta = one_vs_all(data['X'], data['y'], 10, 1)
y_pred = predict_all(data['X'], all_theta)
from sklearn.metrics import classification_report
print(classification_report(data['y'], y_pred))


#2.神经网络的前馈环节（反向梯度见下一任务）
#直接读取已经训练好的weight来处理数据，相当于模型的使用阶段
weight =scio.loadmat("usedata/ex3weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']
X2 = np.matrix(np.insert(data['X'], 0, values=np.ones(pic.shape[0]), axis=1))
y2 = np.matrix(data['y'])
a1 = X2
z2 = a1 * theta1.T
a2 = sigmoid(z2)
a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
z3 = a2 * theta2.T
a3 = sigmoid(z3)
y_pred2 = np.argmax(a3, axis=1) + 1
from sklearn.metrics import classification_report
print(classification_report(y2, y_pred2))