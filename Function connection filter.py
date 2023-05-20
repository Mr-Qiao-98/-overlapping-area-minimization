from sklearn.model_selection import KFold
import numpy as np
import scipy.stats
# import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import xlrd
from sklearn import preprocessing
from decimal import *
import openpyxl 
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
def SX(a,b,c,d,e):
    if a<b and c<d:
        if a<=e<b:
            e = e
    elif b<a and d<c:
        if b<=e<a:
            e = e
    elif a>b and d>c:
        if b<=e<a and c<e<=d:
            e = e
    elif a<b and d<c:
        if a<=e<b and d<e<=c:
            e = e
    else:
        e = 0
    return e
def bubbleSort(arr):
    for i in range(1,len(arr)):
        for j in range(0,len(arr)-i):
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]
    return arr
def countingSort(arr, maxValue):
    bucketLen = maxValue+1
    bucket = [0]*bucketLen
    sortedIndex =0
    arrLen = len(arr)
    for i in range(arrLen):
        if not bucket[arr[i]]:
            bucket[arr[i]]=0
        bucket[arr[i]]+=1
    for j in range(bucketLen):
        while bucket[j]>0:
            arr[sortedIndex] = j
            sortedIndex+=1
            bucket[j]-=1
    return arr
def write_to_excel(path: str, sheetStr, data):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheetStr
    for row_index, row_item in enumerate(data):
        for col_index, col_item in enumerate(row_item):
            sheet.cell(row=row_index+1,column= col_index+1,value=col_item)
    workbook.save(path)
AD = []
CN = []
datas = {} 
for i in range():
    id = str(i)
    df = ""
    fdf = pd.read_excel(df, header=None)
    datas[i] = fdf.values
for i in range():
    id = str(i)
    df = ""
    fdf = pd.read_excel(df, header=None)
    datas[i+95] = fdf.values
# print(datas[1][2][3])
for i in range(90):
    for j in range(90):
        if i<j:
            for k in range(1,191):
#             print(datas[k][i][j])
            
                if k < 96:
                    AD.append(datas[k][i][j])
                else:
                    CN.append(datas[k][i][j])
            
#             AD = bubbleSort(AD)
#             else:
#                 
#                 continue
        else:
            datas[k][i][j] = datas[k][j][i]
            continue
        ADMAX = max(AD)
        ADMIN = min(AD)
#             countingSort(AD, ADMAX)
#             CN = bubbleSort(CN)
        CNMAX = max(CN)
        CNMIN = min(CN)
#             countingSort(CN, CNMAX)      
        for k in range(1,191):
            datas[k][i][j] = SX(CNMIN,ADMIN,CNMAX,ADMAX,datas[k][i][j])
#         else:
#             for key in datas:
#                 datas[key][i][j] = datas[key][j][i]
for k in range(1,191):
    if k < 96:
        path = r''%
        sheetStr = '连接矩阵'
        writeData = datas[k]
        write_to_excel(path, sheetStr, writeData)
    else:
        path = r''''
        # 数据结构1Excel 中sheet 的名字
        sheetStr = '连接矩阵'
        # 数据结构1数据
        writeData = datas[k]
        # 执行
        write_to_excel(path, sheetStr, writeData)