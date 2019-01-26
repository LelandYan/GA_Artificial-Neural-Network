# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/26 8:05'

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics


def QuadraticDiscriminantAnalysis_model(data, result):
    train_x, test_x, train_y, test_y = train_test_split(data, result, test_size=0.3)
    clf = QuadraticDiscriminantAnalysis()
    try:
        clf.fit(train_x, train_y.flatten())
        # expected = test_y
        predicted = clf.predict(test_x)
        # y_hat = clf.predict(x_train)
        # print(clf.score(x_test, y_test))
        # return np.abs(np.average(expected - predicted))
        return metrics.accuracy_score(test_y, predicted)
    except Exception:
        return 0

