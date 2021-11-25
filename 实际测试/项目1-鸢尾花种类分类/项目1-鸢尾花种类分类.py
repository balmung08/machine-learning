import sklearn
from sklearn import datasets  #数据集关系变了，需要单独引用datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn import model_selection
from sklearn import neighbors
import jieba
from scipy import stats
from sklearn import decomposition
import pandas

#KNN算法/K近邻算法
iris = datasets.load_iris()
print(iris)
x_train,x_test,y_train,y_test = model_selection.train_test_split(iris.data,iris.target)#划分训练与测试集
#标准化
transfer = preprocessing.StandardScaler()
x_train = transfer.fit_transform(x_train)   #此处的fit表示按自己的数据计算标准差等数据指标
#test标准化所用的标准差应该与train相同，因此不能加fit
x_test = transfer.transform(x_test)
#KNN预估器
estimator = neighbors.KNeighborsClassifier()#此括号的值表示按最近的k个值里种类偏多的一种分类
#优化模型部分
param_dict={"n_neighbors":[1,3,5,7,9,11]}
estimator = model_selection.GridSearchCV(estimator,param_grid=param_dict,cv=10)#预估器，要调的参数，交叉验证次数

estimator.fit(x_train,y_train)
y_predict = estimator.predict(x_test)
print(y_predict) #预测结果
print(y_predict==y_test) #预测正确情况
score = estimator.score(x_test,y_test)
print(score)#准确率
print(estimator.best_params_)#最佳参数
print(estimator.best_score_)#最佳准确率，是在训练集里二次划分验证的结果，一般高于实际值
print(estimator.best_estimator_)#最佳估计器estimator
print(estimator.cv_results_)#最佳交叉验证结果

