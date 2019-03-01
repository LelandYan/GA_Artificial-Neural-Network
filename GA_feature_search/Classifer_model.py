# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/1 9:36'


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics


def Classifer_model(data, result,model):
    train_x, test_x, train_y, test_y = train_test_split(data, result, test_size=0.3)
    clf = model()
    try:
        clf.fit(train_x, train_y.flatten())
        return clf.score(test_x,test_y)
    except Exception:
        return 0
