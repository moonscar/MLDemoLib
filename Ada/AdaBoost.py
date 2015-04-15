# coding: utf-8
from numpy import *
from collections import namedtuple

decision_dump = namedtuple('decision_dump', 'alpha threshold lt dim')


def weight_stump(X, y, w):
    N, M = X.shape
    best_threshold = 0.0
    min_error = float("inf")
    dimension = -1
    less_than = 1
    for d in xrange(M):
        cur_column = sorted(set(X[:, d]))
        for i in xrange(len(cur_column)-1):
            for j in [1, -1]:
                threshold = (cur_column[i] + cur_column[i+1])/2.0
                error = dot(w, ((j * sign(X[:, d] - threshold).reshape((N, 1))) == y))
                if error < min_error:
                    min_error = error
                    best_threshold = threshold
                    dimension = d
                    less_than = j
    return best_threshold, dimension, min_error, less_than


class Adaboost(object):
    """docstring for Adaboost"""
    def __init__(self, error=0.1):  
        self.e = error

    def train(self, X, y):
        iteration = 50

        N, M = X.shape
        w = ones((1, N))/N
        self.agg = []
        delta_e = N
        Ein = N

        for i in xrange(iteration):
            threshold, dim, error, lt = weight_stump(X, y, w)
            alpha = log((1-error)/error)/2
            
            self.agg.append(decision_dump(alpha, threshold, lt, dim))
            Ein = sum(self.predict(X) == y)
            if Ein < delta_e:
                delta_e = Ein
            else:
                self.agg.pop()
                break

            # update w
            expon = alpha * ((self.predict(X)).reshape((N, 1)) * y)
            w = w * exp(expon).T
            w /= sum(w)

    def predict(self, X):
        dicision = 0.0
        for a in self.agg:
            dicision += a.alpha*sign(X[:, a.dim]-a.threshold)*a.lt
        return sign(dicision).reshape((X.shape[0], 1))

if __name__ == '__main__':
    X = array(range(10))
    y = array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

    X = X.reshape((10, 1))
    y = y.reshape((10, 1))
    clf = Adaboost()
    clf.train(X, y)
