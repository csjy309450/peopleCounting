# -*-encoding=utf-8-*-

import time
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn import svm
from sklearn.externals import joblib


datafilePath = (
	"/home/yangzheng/testData/pet/View_001_lable/frame_0000.npy",
	"/home/yangzheng/testData/pet/View_001_lable/frame_0020.npy",
	"/home/yangzheng/testData/pet/View_001_lable/frame_0037.npy",
	"/home/yangzheng/testData/pet/View_001_lable/frame_0053.npy",
	"/home/yangzheng/testData/pet/View_001_lable/frame_0059.npy",
	"/home/yangzheng/testData/pet/View_001_lable/frame_0114.npy",
    "/home/yangzheng/testData/pet/View_001_lable/frame_0214.npy",
)
_xlim = (0, 768)
_ylim = (0, 576)
threshold_ = 0.2

def loadDataset(_filePaths):
    """
    load data set
    :param _dataset: 数据存放的darray对象
    :param filePaths: 存放.npy文件path的tuple对象
    :return:
    """
    data = np.empty((0, 4))
    for path in _filePaths:
        data = np.row_stack([data, np.load(path)])
        # print 'Load data success\n', data
    return data

def showDatasetDistribution(_dataset, _xlim=None, _ylim=None):
    """
    显示dataset分布情况
    :param _xlim:
    :param _ylim:
    :return:
    """
    import matplotlib.pyplot as plt
    if _xlim is tuple and _ylim is tuple:
        plt.xlim(_xlim)
        plt.ylim(_ylim)
    plt.scatter(_dataset[:, 0], _dataset[:, 1])
    plt.show()

def makeTrainsetAndTestset(_dataset, _threshold=0.3, _bias=False):
    count = _dataset.shape[0]
    test_count = int(count * _threshold)
    # Split the data into training/testing sets
    data_X_train = _dataset[test_count:, 0:2]
    data_X_test = _dataset[0:test_count, 0:2]

    # Split the targets into training/testing sets
    data_y_train = _dataset[test_count:, 2:]
    data_y_test = _dataset[0:test_count, 2:]

    if _bias is True:
        ones_train = np.ones((data_X_train.shape[0]))
        data_X_train = np.column_stack([ones_train, data_X_train])
        ones_test = np.ones((data_X_test.shape[0]))
        data_X_test = np.column_stack([ones_test, data_X_test])
    return [data_X_train, data_y_train], [data_X_test, data_y_test]

def LinearRegression(_dataset, _modePath=''):
    """
    linear regession
    """
    data_train, data_test = makeTrainsetAndTestset(_dataset, _threshold=0)
    reg = linear_model.LinearRegression()
    reg.fit(data_train[0], data_train[1])
    joblib.dump(reg, _modePath+'personTall_Model.m')

if __name__ == '__main__':
    dataset = loadDataset(datafilePath)
    LinearRegression(dataset)



# pred = reg.predict(data_X_test)
# print "linear regession\n", reg.score(data_X_test, data_y_test)
# # test
# for i in xrange(0, pred.shape[0]):
#     print pred[i], data_y_test[i]
# print '\n'

# """
# Ridge regession
# """
# reg = linear_model.Ridge(alpha = .5)
# reg.fit(data[:, 0:2], data[:, 2])
# pred = reg.predict(data_X_test)
# print "Ridge regession\n", reg.score(data_X_test, data_y_test)
# # test
# for i in xrange(0, pred.shape[0]):
#     print pred[i], data_y_test[i]
# print '\n'
#
# """
# Lasso regession
# """
# reg = linear_model.Lasso(alpha = 0.1)
# reg.fit(data[:, 0:2], data[:, 2])
# pred = reg.predict(data_X_test)
# print "Lasso regession\n", reg.score(data_X_test, data_y_test)
# # test
# for i in xrange(0, pred.shape[0]):
#     print pred[i], data_y_test[i]
# print '\n'
#
# """
# LARS Lasso regession
# """
# reg = linear_model.LassoLars(alpha=.1)
# reg.fit(data[:, 0:2], data[:, 2])
# pred = reg.predict(data_X_test)
# print "LARS Lasso regession\n", reg.score(data_X_test, data_y_test)
# # test
# for i in xrange(0, pred.shape[0]):
#     print pred[i], data_y_test[i]
# print '\n'
#
# """
# Kernel ridge regression.
# """
# clf = KernelRidge(alpha=1)
# clf.fit(data[:, 0:2], data[:, 2])
# pred = clf.predict(data_X_test)
# print "Kernel ridge regression.\n", clf.score(data_X_test, data_y_test)
# # test
# for i in xrange(0, pred.shape[0]):
#     print pred[i], data_y_test[i]
# print '\n'
#
# """
# SVM regression.
# """
# clf = svm.SVR()
# clf.fit(data[:, 0:2], data[:, 2])
# pred = clf.predict(data_X_test)
# print "SVM regression.\n", clf.score(data_X_test, data_y_test)
# # test
# for i in xrange(0, pred.shape[0]):
#     print pred[i], data_y_test[i]
# print '\n'