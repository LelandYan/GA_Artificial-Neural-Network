# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/20 21:39'
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


def svm_model(data, result):
    train_x, test_x, train_y, test_y = train_test_split(data, result, test_size=0.3)
    clf = SVC()
    clf.fit(train_x, train_y)
    #expected = test_y
    #predicted = clf.predict(test_x)
    # y_hat = clf.predict(x_train)
    #print(clf.score(x_test, y_test))
    # return np.abs(np.average(expected - predicted))
    return clf.score(test_x,test_y)

if __name__ == '__main__':
    CSV_FILE_PATH = 'parkinsons.csv'
    # read the file
    df = pd.read_csv(CSV_FILE_PATH)
    shapes = df.values.shape
    # the eigenvalue of file
    input_data = df.values[:, 1:shapes[1] - 1]
    # the result of file
    result = df.values[:, shapes[1] - 1:shapes[1]]
    print(svm_model(input_data, result))
