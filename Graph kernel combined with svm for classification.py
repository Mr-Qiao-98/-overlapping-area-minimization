from sklearn.model_selection import KFold
import numpy as np
import scipy.stats
# import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import xlrd
from sklearn import preprocessing
from decimal import *
import openpyxl # openpyxl引入模块
from grakel import Graph
from time import time
from sympy import DiracDelta
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
from sklearn import preprocessing
from scipy import stats
from grakel.kernels import WeisfeilerLehmanOptimalAssignment
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.kernels import OddSth
from grakel.kernels import ShortestPath
from grakel.kernels import PyramidMatch
from grakel.kernels import NeighborhoodHash

#输入数据推荐使用numpy数组，使用list格式输入会报错
def K_Flod_spilt(K,fold,data,label):
    '''
    :param K: 要把数据集分成的份数。如十次十折取K=10
    :param fold: 要取第几折的数据。如要取第5折则 flod=5
    :param data: 需要分块的数据
    :param label: 对应的需要分块标签
    :return: 对应折的训练集、测试集和对应的标签
    '''
    split_list = []    #定义一个列表
    kf = KFold(n_splits=K,shuffle=True)
    #定义几折
    for train, test in kf.split(data):    #训练测试在数据中循环
        split_list.append(train.tolist())
        split_list.append(test.tolist())
    train,test=split_list[2 * fold],split_list[2 * fold + 1]
    return  data[train], data[test], label[train], label[test]  #已经分好块的数据集
for l in range(1,11):
    datas = {} 
#     for i in range(0,100):
#         id = str(i)
#         df = "E:/数据处理/连接矩阵2/"+"text"+id+ ".xlsx"
#         fdf = pd.read_excel(df, header=None)
#         datas[i] = fdf.values
    for i in range(1,51):
        id = str(i)
        df = "E:/数据处理/JS散度2/连接矩阵%d/"%l+"AD"+id+ ".xlsx"
        fdf = pd.read_excel(df, header=None)
        datas[i] = fdf.values
    for i in range(1,51):
        id = str(i)
        df = "E:/数据处理/JS散度2/连接矩阵%d/"%l+"CN"+id+ ".xlsx"
        fdf = pd.read_excel(df, header=None)
        datas[i+50] = fdf.values
    # print(datas)
    data = xlrd.open_workbook("E:/数据处理/连接矩阵1/标签文件/图标签2.xlsx") #读取文件
    table = data.sheet_by_index(0) #按索引获取工作表，0就是工作表1
    y = table.row_values(0) #读取每行数据，保存在line里面，line是list
    graphs = list()  #图的列表
    for key in datas:
        edges = list()
        edge_labels = dict()
        for i in range(90):
            for j in range(90):
                if datas[key][i][j] > 0:
                    edges.append((i,j))
                    edge_labels[(i,j)] = 1
    #     for i in range(8):
        node_labels = dict()
        for j in range(90):
            node_labels[j] = j+1
        graphs.append(Graph(edges, node_labels = node_labels, edge_labels = edge_labels))
    y = np.array(y)
    # print(graphs)
    # print(y)
    print(l)
    K=10
    graphs = np.array(graphs)
    accs = []
    for fold in range(10):
        x1,x2,y1,y2 = K_Flod_spilt(K,fold,graphs,y)
    
    #     print(y1)
    #     print('x2')
    #     print(x2)
#         gk1 = WeisfeilerLehmanOptimalAssignment(n_jobs = None , verbose = False , normalize = False , n_iter = 6 , sparse = False)
#         gk2 = WeisfeilerLehman(n_iter=i, base_graph_kernel = VertexHistogram, normalize=True)
#         gk3 = OddSth(n_jobs=None, normalize=False, verbose=False, h=None)
        gk4 = ShortestPath(n_jobs=None, normalize=False, verbose=False, with_labels=True, algorithm_type='floyd_warshall')
#         gk5 = PyramidMatch(n_jobs=None, normalize=False, verbose=False, with_labels=True, L=4, d=6)
#         gk1 = NeighborhoodHash(n_jobs=None, normalize=False, verbose=False, random_state=None, R=3, nh_type='simple', bits=8)
#         K_train1 = gk1.fit_transform(x1)
#         K_test1 = gk1.transform(x2)
#         K_train2 = gk2.fit_transform(x1)
#         K_test2 = gk2.transform(x2)
#         K_train3 = gk3.fit_transform(x1)
#         K_test3 = gk3.transform(x2)
        K_train4 = gk4.fit_transform(x1)    #x1是构造的图
#         print(K_train4[89][89])
#         print(K_train4)
        K_test4 = gk4.transform(x2)
#         K_train5 = gk5.fit_transform(x1)
#         K_test5 = gk5.transform(x2)
#         print("对比1")
#         print(K_test1)
#         print(K_test1[0][0])
#         print(K_test1[0][0]/10)
#         print(K_test1[0][1]/10)
#         print(K_test2[0])
#         print(K_test3[0])
#         print(K_test4[0])
#         print(K_test5[0])
        
#         print("对比2")
#         print(K_test1[1][1])
#         print(K_test2)
#         print(K_test3[1])
#         print(K_test4[1])
#         print(K_test5[1])
#         for i in range(10):
#             for j in range(90):
# #                 K_train3[i][j] = (K_train1[i][j]+K_train2[i][j])/2
#                 K_test3[i][j] = (K_test1[i][j]+K_test2[i][j])/2
#         print("对比3")
#         print(K_test3)
        clf = SVC(kernel='precomputed')
        clf.fit(K_train4, y1)     #图与标签匹配并进行训练
        y_pred = clf.predict(K_test4)
#         print(y2)
#         print(y_pred)
#     y_pred = y_pred*(-1)
#         print("Classification accuracy: %0.2f" % accuracy_score(y2, y_pred))
        accs.append(accuracy_score(y2, y_pred))
    print(accs)   #列出所有结果
