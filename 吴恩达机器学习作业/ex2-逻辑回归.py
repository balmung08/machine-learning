#1.逻辑回归
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("usedata/ex2data1.txt",header=None, names=['data 1', 'data 2', 'result'])
data_0 = data[data['result'] == 0]
data_1 = data[data['result'] == 1]



#数据可视化
fig, ax = plt.subplots()
ax.scatter(data_0['data 1'], data_0['data 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(data_1['data 1'], data_1['data 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set(xlabel='data 1 Score',ylabel='data 2 Score')


#方法一：梯度下降
#非线性激活函数
def sigmoid(x):
    result = 1/(1+np.exp(-x))
    return result

#逻辑回归的代价函数(仅在二分类成立)
def costfunction(theta, x, y):
    theta = np.matrix(theta)
    X = np.matrix(x)
    y = np.matrix(y)
    #注意sigmoid函数和log项
    first = np.multiply(y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    result = -np.sum(first + second) / len(x)
    return result

data.insert(0, 'zero', 1)
x = data.iloc[:,0:3]
y = data.iloc[:,3:4]
theta = np.zeros(3)
#预先转为矩阵
x = np.matrix(np.array(x))
y = np.matrix(np.array(y))


#梯度计算公式，用于梯度下降
def gradient(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    #读取theta维数
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(x * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error,x[:,i])
        grad[i] = np.sum(term) / len(x)
    return grad

#梯度下降算法实现
def gradientDescent(x, y, theta, alpha, iters):
    theta = np.matrix(theta)
    temp = np.matrix(np.zeros(theta.shape))#过程矩阵
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)#按照迭代次数创造全是0的行矩阵
    for i in range(iters):
        for j in range(parameters):
            temp[0, j] = theta[0, j] - ((alpha * gradient(theta,x,y)[j]))
        theta = temp
        cost[i] = costfunction(theta,x, y)
        #print(cost[i])
    return theta, cost

#theta, cost = gradientDescent(x, y, theta, 0.008, 300000)
#result: 0.008 300000  [[-25.72636892   0.2167094    0.19579284]]
#result: 0.008 400000  [[-25.72695899   0.22163357   0.19844202]]

#方法二：调用scipy的api
import scipy.optimize as opt
result = opt.fmin_tnc(func=costfunction, x0=theta, fprime=gradient, args=(x, y))

#方法三：调用sklearn的api
from sklearn import linear_model
reg = linear_model.LogisticRegression()
reg.fit(x,y)


print("梯度下降参数0.008 400000结果：[-25.72695899   0.22163357   0.19844202]")
print("scipy-cost:", costfunction(result[0], x, y))
print("scipy-最优θ组:", result[0])
print("sklearn-最优θ组:", reg.intercept_[0], reg.coef_[0][1], reg.coef_[0][2])

#结果可视化
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(data_0['data 1'], data_0['data 2'], s=50, c='b', marker='o', label='Admitted')
ax1.scatter(data_1['data 1'], data_1['data 2'], s=50, c='r', marker='x', label='Not Admitted')
x = np.linspace(30, 100, 100)
y = (25.72695899 - 0.22163357 * x)/0.19844202
ax1.plot(x, y, color='blue', linewidth=0.5, label="1")#s表示标记的大小
y = (-result[0][0] - result[0][1] * x)/result[0][2]
ax1.plot(x, y, color='orange', linewidth=0.5, label="2")#s表示标记的大小
y = (- reg.intercept_[0] - reg.coef_[0][1] * x)/reg.coef_[0][2]
ax1.plot(x, y, color='black', linewidth=0.5, label="3")#s表示标记的大小
ax1.set(xlabel='data 1 Score', ylabel='data 2 Score')
ax1.legend()


#模型的评价
def predict(theta, x):
    probability = sigmoid(x * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def correct(m):
    x = data.iloc[:,0:3]
    x = np.matrix(x)
    y = data.iloc[:,3:4]
    y = np.array(y).ravel()
    theta = np.matrix(m)
    result = predict(theta,x)
    correct = 0
    for i in range(len(y)):
        if y[i] == result[i]:
            correct += 1
    return correct/len(y)

print("梯度下降准确率：",correct([-25.72695899,0.22163357,0.19844202]))
print("scipy准确率：",correct(result[0]))
print("sklearn准确率：",correct([reg.intercept_[0],reg.coef_[0][1], reg.coef_[0][2]]))

#2.正则化逻辑回归（非线性）
print("-------------------------------------")
data = pd.read_csv("usedata/ex2data2.txt",header=None, names=['data 1', 'data 2', 'result'])
data_0 = data[data['result'] == 0]
data_1 = data[data['result'] == 1]

#数据可视化
fig, ax = plt.subplots()
ax.scatter(data_0['data 1'], data_0['data 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(data_1['data 1'], data_1['data 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set(xlabel='data 1 Score',ylabel='data 2 Score')
#方法一：增加维数梯度下降
#观察知此数据集不能像之前一样使用线性的函数来分割，而逻辑回归只适用于线性的分割，故此数据集不适合直接使用逻辑回归。
#更好的使用数据集的方式是为每组数据创造更多的特征。所以我们要为每组数据添加最高到6次幂的特征，即创造六次多项式作为多项式回归
data.insert(3, 'Ones', 1)
for i in range(1, 7):  #六维
    for j in range(0, i+1):
        data['F' + str(i-j) + str(j)] = np.power(data['data 1'], i-j) * np.power(data['data 2'], j)
        #按照维数，F13即表示data1一次项乘data2三次项
#删除原本的index
data.drop('data 1', axis=1, inplace=True)
data.drop('data 2', axis=1, inplace=True)
x = data.iloc[:,1:29]
y = data.iloc[:,0:1]
theta = np.zeros(28)
#预先转为矩阵
x = np.matrix(np.array(x))
y = np.matrix(np.array(y))
theta_m1, cost = gradientDescent(x, y, theta, 30, 1000)

theta_test=np.array(theta_m1).ravel()

#数据正确率评估
def predict(theta, x):
    probability = sigmoid(x * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def correct(m):
    x = data.iloc[:, 1:29]
    x = np.matrix(np.array(x))
    y = data.iloc[:, 0:1]
    y = np.array(y).ravel()
    theta = np.matrix(m)
    result = predict(theta,x)
    correct = 0
    for i in range(len(y)):
        if y[i] == result[i]:
            correct += 1
    return correct/len(y)
print(theta_test)
print("梯度下降准确率：",correct(theta_test))

#方法二：调用scipy的api
import scipy.optimize as opt
# 实现正则化的代价函数
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


# 实现正则化的梯度函数
def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])
    return grad

theta_2 = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x, y, 10))
print(theta_2[0])
theta_test2=np.array(theta_2).ravel()
print("scipy准确率：",correct(theta_test2[0]))


#绘制函数分界线
theta_m1=np.array(theta_m1)
def hfunc2(theta, x1, x2):
    #用高维特征计算出函数结果
    temp = theta[0][0]
    place = 0
    for i in range(1, 7):
        for j in range(0, i+1):
            temp+= np.power(x1, i-j) * np.power(x2, j) * theta[0][place+1]
            place+=1
    return temp
t1 = np.linspace(-1.5, 1.0, 2000)
t2 = np.linspace(-1.5, 1.0, 2000)
cordinates = [(x, y) for x in t1 for y in t2]
x_cord, y_cord = zip(*cordinates)
h_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
h_val['hval'] = hfunc2(theta_m1, h_val['x1'], h_val['x2'])#此处更改填入的数据
decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
x, y = decision.x1,decision.x2
ax.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()