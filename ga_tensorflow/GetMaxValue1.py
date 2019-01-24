# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/24 10:05'
import pandas as pd
import numpy as np
import random as rd
from sklearn import neighbors
from sklearn import metrics
from ga_tensorflow.svm_model import *

numAtt = 0.0
# 样本数量
N = 0
# 定义DNA位数
M = 0
# 定义样本DNA数组
DNA = 0
# 分类精确度
show = []
# 定义样本存活的累计概率0-1
alive = []
# 定义淘汰的阈值
sigma = 3
# 定义繁衍的代数
T = 200
# 交叉的概率
interPro = 0.8
# 突变的概率
breakPro = 0.1
pro = 0
str_1 = ""
diseaseName = ["SRBCT", "MLL", "Lymphoma"]
S = [10, 10, 10]
prob = [0.02, 0.006, 0.005]
CSV_FILE_PATH = 'csv_result-ALL-AML_train.csv'
# read the file
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
# the eigenvalue of file
input_data = df.values[:, 1:shapes[1] - 1]
# the result of file
result = df.values[:, shapes[1] - 1:shapes[1]]
# the length of eigenvalue
value_len = input_data.shape[1]
# the length of result
pop_len = result.shape[0]
# DNA length
DNA_SIZE = value_len
# population size
POP_SIZE = pop_len


# 初始化函数
def init():
    N = pop_len
    M = DNA_SIZE
    DNA = np.zeros((N, M))
    show = [None for i in range(N)]
    for i in range(N):
        for j in range(M):
            if rd.random < pro:
                DNA[i, j] = 1
        show[i] = f(i)
    alive = [None for i in range(N)]


def translateDNA(pop):
    index_list = []
    for i in range(len(pop)):
        if pop[i] == 1:
            index_list.append(i)
    return index_list


# 适应度函数
def f(x):
    n = 0
    for i in range(len(DNA[x, :])):
        if (DNA[x, i] == 1): n += 1
    data = input_data[:, translateDNA(DNA[x, :])]
    return svm_model(data, result) * 0.8 + 0.2 * (1 - n / DNA_SIZE)


# 求累计存活概率函数
def icanlive():
    sum = 0
    alivesum = 0
    for i in range(N):
        sum += show[i]
    for i in range(N):
        alivesum += show[i] / sum
        alive[i] = alivesum


# 淘汰函数
def dieout():
    times = [None for i in range(N)]
    for i in range(N):
        times[i] = 0
    # 遗传过程
    for i in range(2 * N):
        myro = rd.randint(100) / 100
        for j in range(N):
            if myro < alive[j]:
                if j != 0 and myro > alive[j - 1]:
                    times[j] += 1
                    break
                elif j == 0:
                    times[j] += 1
                    break
    # 找出本次遗传中最强的种族
    max = times[0]
    sign = 0
    for k in range(N):
        if times[k] < sigma:
            max = times[k]
            sign = k
    for k in range(N):
        if times[k] < sigma:
            show[k] = show[sign]
            for i in range(M):
                DNA[k, i] = DNA[sign, i]

