from numpy import *
from collections import namedtuple

decision_dump = namedtuple('decision_dump', 'alpha threshold dim')

def weight_stump(X, y, w):
    N, M = X.shape
    best_threshold = 0.0
    min_error = float("inf")
    dimension = -1
    for d in xrange(M):
        cur_column = sorted(X[:, d])
        for i in xrange(len(cur_column)-1):
            threshold = (cur_column[i] + cur_column[i+1])/2.0
            error = dot(w, (sign(X[:, d] - threshold) == y))
            if error < min_error:
                min_error = error
                best_threshold = threshold
                dimension = d
    return best_threshold, dimension, min_error


class Adaboost(object):
    """docstring for Adaboost"""
    def __init__(self):
        return

    def train(self, X, y):
        iteration = 50

        N, M = X.shape
        w = ones((1, N))/N
        aggregation = []
        
        for i in xrange(iteration):
            threshold, dim, error = weight_stump(X, y, w)
            delta_w = sqrt((1-error)/error)
            alpha = log(delta_w)
            # update w
            aggregation.append(decision_dump(alpha, threshold, dim))

        self.agg = aggregation

    def predict(self, X):
        dicision = 0.0
        for a in self.agg:
            dicision += a.alpha*sign(X[a.dim]-a.threshold)
        return sign(dicision)

if __name__ == '__main__':
    pass