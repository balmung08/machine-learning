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
''''
#数据集加载
datasets.load_# 加载本地小数据集
datasets.fetch_#网上下载大数据集 fetch括号里写个subset='all'
返回值为bunch类型，是字典和数组的复合类型，有data/target/target_names/feature_names等键值对，target的值对应name栏里的项
独有调用格式如bunch.data，也可以按字典格式调用
bunch.DESCR是数据集的描述 

#数据集的划分，一般70%-80%训练，20%-30%验证
model_selection.train_test_split(bunch.data,bunch.target,test_size=0.2,random_state=22)#0.2表示比例划分
返回四个值，即为train与test的data与target///x_train,y_train,x_test,y_test

#特征提取-字典特征提取
one-hot编码，完全公平的01表示
def dict_demo():
    data = [{"city":"beijing","temperature":100},{"city":"shanghai","temperature":60},{"city":"yichang","temperature":33}]
    transfer = feature_extraction.DictVectorizer(sparse=False)#sparse表示为非0数据坐标+数据值/false时为直观的矩阵
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names())
    print(data_new)
#特征提取-文本特征提取
#①统计样本特征词的出现个数
def count_demo():
    data = ["life is short, i use python","i will draft you one"]
    transfer = feature_extraction.text.CountVectorizer()#空里可以填stop_words=[],可以停用这些词语使其不参与统计
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names())
    print(data_new.toarray())
#②统计中文文本，需要把词语之间打空格，使用程序自动分词
def cut_words(text):
    text = " ".join((jieba.cut(text)))
    print(text)
    return text
def count_chinese():
    data = {"弃我去者昨日之日不可留","乱我心者今日之日多烦忧"}
    data_new = []
    for text in data:
        data_new.append(cut_words(text))
    transfer = feature_extraction.text.CountVectorizer()#空里可以填stop_words=[],可以停用这些词语使其不参与统计
    data_final = transfer.fit_transform(data_new)
    print(transfer.get_feature_names())
    print(data_final.toarray())
#③统计文本，打印各种词语的重要程度指数
def cut_words(text):
    text = " ".join((jieba.cut(text)))
    print(text)
    return text
def tfidf():
    data = {"70年前，由中华优秀儿女组成的中国人民志愿军肩负着人民的重托、民族的期望，高举保卫和平、反抗侵略的正义旗帜，跨过鸭绿江，同朝鲜人民和军队一道，历经两年零9个"}
    for text in data:
        data_new.append(cut_words(text))
    transfer = feature_extraction.text.TfidfVectorizer()  # 空里可以填stop_words=[],可以停用这些词语使其不参与统计
    data_final = transfer.fit_transform(data_new)
    print(transfer.get_feature_names())
    print(data_final.toarray())
#特征预处理-归一化：x'=x-xmin/xmax-xmin    (x'' = x'(mx-mi)+mi)
data = [[20,30,40],
        [60,60,90],
        [40,80,70]]
transfer = preprocessing.MinMaxScaler(feature_range=[2,3])
data_new = transfer.fit_transform(data)
print(data_new)
#特征预处理-标准化 如果有异常值归一化效果会很差 x' = （x-x均）/σ（标准差）
data = [[20,30,40],
        [60,60,90],
        [40,80,70]]
transfer = preprocessing.StandardScaler()
data_new = transfer.fit_transform(data)
print(data_new)
#降维 维度即为数组嵌套的层数 对二维数组降低随机变量的个数，得到一组互不相干的主变量
#降维-特征选择-filter过滤式-方差选择法（低方差特征过滤）
def demo():
    data = []
    transfer = sklearn.feature_selection.VarianceThreshold(threshold=5)
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names())
    print(data_new.toarray())
#降维-特征选择-filter过滤式-相关系数法（高相关系数过滤）
#相关系数范围属于-1到1 0.4/0.7/1三档
def demo():
    data = [[20, 77, 40, 78],
            [24, 58, 20, 21],
            [98, 20, 40, 29]]
    r = stats.pearsonr(data[1],data[2])#返回的第一个数是相关系数，第二个是显著性
    print(r)
    #相关性高的选一个作为代表或者加权求和作为一个新特征或主成分分析
    transfer = decomposition.PCA(n_components=3)#整数：减少到多少特征 小数：保留百分之多少的信息
    data_new = transfer.fit_transform(data)
    print(data_new)

#KNN算法/K近邻算法
iris = datasets.load_iris()
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
'''






