# coding: utf-8
from numpy import *
from MLDemoLib.AdaBoost import *

def loadSimpleData():
    dataMat = array([[1. , 2.1],
        [2. , 1.1],
        [1.3 , 1.],
        [1. , 1.],
        [2. , 1.]])
    classLabels = array([1.0,1.0,-1.0,-1.0,1.0])
    return dataMat, classLabels


def loaddata(fname):
    fh = open(fname)
    data = []
    label = []
    for l in fh.readlines():
        row = [float(i) for i in l.split()]
        data.append(row[:-1])
        label.append(row[-1])
    X = array(data)
    y = array(label)
    y[y==0] = -1
    return X, y.reshape((X.shape[0], 1))


def valiation(X, y, clf):
    N, M = X.shape
    error = 0.0
    error = sum(clf.predict(X) == y)
    error /= float(N)
    return error


def test_main():
    data, label = loaddata('horseColicTraining.txt')
    clf = Adaboost()
    clf.train(data, label)

    v_data, v_lable = loaddata('horseColicTest.txt')
    print valiation(v_data, v_lable, clf)

if __name__ == '__main__':
    test_main()