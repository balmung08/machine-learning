#1.输出一个5*5的单位矩阵
import numpy as np
a = np.zeros((4,4),int)

#2 单变量的线性回归（根据变量1预测变量2）
import pandas as pd
import matplotlib.pyplot as plt

#方法一：自己造轮子
#数据预处理
data = pd.read_csv("usedata/ex1data1.txt",names=['Population', 'Profit'])
fig = plt.figure()
ax1 = fig.add_subplot(211)#划分为x行x列，在第x个位置绘图
ax2 = fig.add_subplot(212)#划分为x行x列，在第x个位置绘图
ax1.set(title='method1',ylabel='y', xlabel='x')#轴范围，轴标题，图标题
ax1.scatter(x=data.Population, y=data.Profit, s=5,color='red', marker='.')#s表示标记的大小
ax2.set(title='method2',ylabel='y', xlabel='x')#轴范围，轴标题，图标题
ax2.scatter(x=data.Population, y=data.Profit, s=5,color='red', marker='.')#s表示标记的大小
#插入一列1用于乘θ0，表示表达式中的常量项
data.insert(0, 'Ones', 1)
x = data.iloc[:,:-1]#iloc a:b,c:d--a到b行，c到d列，前闭后开
y = data.iloc[:,2:3]
x = np.matrix(x)
y = np.matrix(y)
theta = np.matrix(np.array([0,0]),dtype=float)#只有矩阵才能写转置

#计算代价函数J(Ѳ)
def costfunction(x,y,theta):
    temp = np.power(((x * theta.T) - y), 2)
    result = np.sum(temp)/(2*len(x))
    return result
#梯度下降算法实现
def gradientDescent(x, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))#过程矩阵
    parameters = int(theta.shape[1])#读取theta的个数
    cost = np.zeros(iters)#按照迭代次数创造全是0的行矩阵
    for i in range(iters):
        #error修正到theta矩阵里
        error = (x * theta.T) - y
        for j in range(parameters):
            #将error乘到矩阵x上就直接是代价函数的导数
            term = np.multiply(error, x[:,j])
            #对temp暂存量作更新，即新值等于原值减导数与lr的乘积
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))
        theta = temp
        cost[i] = costfunction(x, y, theta)
    return theta, cost

alpha = 0.01#学习率
iters = 3000#迭代次数

theta_result, cost = gradientDescent(x, y, theta, alpha, iters)
x = data.Population
y = data.Population*theta_result[0,1]+theta_result[0,0]
ax1.plot(x,y,color='blue', linewidth=0.5)#s表示标记的大小


#方法二：使用sklearn的api
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(np.array(data.Population).reshape(-1,1), data.Profit)
x = data.Population
y = reg.predict(np.array(data.Population).reshape(-1,1))
print(reg.coef_[0],theta_result[0,1])
print(reg.intercept_,theta_result[0,0])
ax2.plot(x,y,color='green', linewidth=0.5)#s表示标记的大小


#损失函数J(θ)的可视化
fig1 = plt.figure()
ax3 = fig1.add_subplot(111, projection='3d')
ax3.set(title='visual',ylabel='θ1', xlabel='θ0',zlabel='J(θ)')#轴范围，轴标题，图标题
x = data.iloc[:,:-1]#iloc a:b,c:d--a到b行，c到d列，前闭后开
y = data.iloc[:,2:3]
x = np.matrix(x)
y = np.matrix(y)
o = np.arange(-4, 4, 0.05)
p = np.arange(-4, 4, 0.05)
o, p = np.meshgrid(o, p)    # x-y 平面的网格
R = np.ones((160,160))
for i in range(0,160):
    for m in range(0,160):
        R[i,m] = costfunction(x,y,np.matrix([[o[i][m], p[i][m]]]))
ax3.plot_surface(o,p,R, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax3.contourf(o,p,R,zdir='z',offset=-2)
ax3.set_zlim(0,800)
plt.show()


#3.多变量线性回归（根据变量1，2预测变量3）
#数据预处理
print("-------------------------------------")
data_n = pd.read_csv("usedata/ex1data2.txt",names=['data1', 'data2','result'])

fig1 = plt.figure()
ax4 = fig1.add_subplot(211)#划分为x行x列，在第x个位置绘图
ax5 = fig1.add_subplot(212)#划分为x行x列，在第x个位置绘图
ax4.set(title='method1',ylabel='y', xlabel='x')#轴范围，轴标题，图标题
ax4.scatter(x=data.Population, y=data.Profit, s=5,color='red', marker='.')#s表示标记的大小
ax5.set(title='method2',ylabel='y', xlabel='x')#轴范围，轴标题，图标题
ax5.scatter(x=data.Population, y=data.Profit, s=5,color='red', marker='.')#s表示标记的大小



#数据预处理
#观察发现此数据集需要特征归一化
data_n = (data_n - data_n.mean()) / data_n.std()
data_n.insert(0, 'Ones', 1)

x = data_n.iloc[:,:-1]#iloc a:b,c:d--a到b行，c到d列，前闭后开
y = data_n.iloc[:,3:4]
x = np.matrix(x)
y = np.matrix(y)
theta = np.matrix(np.array([0,0,0]),dtype=float)#只有矩阵才能写转置

#方法一：用自己的轮子-梯度下降
alpha = 0.01#学习率
iters = 3000#迭代次数
theta_result, cost = gradientDescent(x, y, theta, alpha, iters)
print("梯度下降：",theta_result[0,0],theta_result[0,1],theta_result[0,2])

#方法二：自己造轮子-标准方程法
def normalEqn(x, y):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.linalg.inv(x.T*x)*x.T*y
    return theta
theta_result = normalEqn(x, y)
print("标准方程：",theta_result[0,0],theta_result[1,0],theta_result[2,0])

#方法三：使用sklearn的api
reg = linear_model.LinearRegression()
reg.fit(x, y)
m = reg.predict(x)
print("api：",reg.intercept_[0],reg.coef_[0][1],reg.coef_[0][2],)
