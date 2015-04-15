# coding: utf-8
from numpy import *


def sigmoid(X):
    return 1.0/(1 + exp(-X))


class LogisticRegression:
    """docstring for LogisticRegression"""
    def __init__(self, C=1.0, max_iter=500, eta=0.001):
        # super(LogisticRegression, self).__init__()
        self._C = C
        self._max_iter = max_iter
        self._eta = eta

    def train(self, X, y):
        N, M = X.shape
        X = hstack((ones((N, 1)), X))

        w = ones((M+1, 1))

        for i in xrange(self._max_iter):
            h = sigmoid(dot(X, w))
            error = (y - h)
            gradient = dot(X.transpose(), error)
            w = w + self._eta * gradient

        self._w = w
        return w

    def predict(self, X):
        return sign(sigmoid(dot(X, self._w)) - 0.5)
