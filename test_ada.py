# coding: utf-8
from numpy import *
import AdaBoost 

def loadSimpleData():
    dataMat = array([[1. , 2.1],
        [2. , 1.1],
        [1.3 , 1.],
        [1. , 1.],
        [2. , 1.]])
    classLabels = array([1.0,1.0,-1.0,-1.0,1.0])
    return dataMat, classLabels

def test():
    X, y = loadSimpleData()
    ada = AdaBoost.Adaboost()
    ada.train(X, y)
    print ada.predict([0, 0])

if __name__ == '__main__':
    test()