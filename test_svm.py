import SVM
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return array(dataMat),array(labelMat)

def test():
    a = SVM.SVM(1)
    X, y = loadDataSet()
    y = y.reshape((X.shape[0], 1))
    a.simTrain(X, y, 0.6, 0.001, 40)

if __name__ == '__main__':
    test()