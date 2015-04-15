# coding: utf-8
from MLDemoLib.LogisticRegression import *
import matplotlib.pyplot as plt
from numpy import *


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
    return X, y.reshape((X.shape[0], 1))


def plot(X, y, w):
    '''Plot scatter and decision bound for 2d data.'''
    N, M = X.shape

    if M > 2:
        print 'Hard to demostrate.'
        return

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in xrange(N):
        if y[i] == 1:
            xcord1.append(X[i, 0])
            ycord1.append(X[i, 1])
        else:
            xcord2.append(X[i, 0])
            ycord2.append(X[i, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = arange(-3.0, 3.0, 0.1)
    y = (-w[0] - w[1]*x)/w[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def valiation(X, y, lr):
    N, M = X.shape
    error = 0.0
    error = sum(lr.predict(X) == y)
    error /= float(N)
    return error


def test_main():
    X, y = loaddata('testSet.txt')
    demo = LogisticRegression()
    demo.train(X, y)
    plot(X, y, demo._w)

    data, label = loaddata('horseColicTraining.txt')
    my_lr = LogisticRegression()
    my_lr.train(data, label)

    v_data, v_lable = loaddata('horseColicTest.txt')
    print valiation(v_data, v_lable, my_lr)


if __name__ == '__main__':
    test_main()
