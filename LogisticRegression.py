
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model


def sigmoid(X):
    return 1.0/(1 + exp(-X))


def load_toy_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 2]
    y = y[y != 2]

    return X, y


class LogisticRegression:
    """docstring for LogisticRegression"""
    def __init__(self, C=1.0, max_iter=500, eta=0.001):
        # super(LogisticRegression, self).__init__()
        self.C = C
        self.max_iter = max_iter
        self.eta = eta

    def train(self, X, y):
        N, M = X.shape

        w = ones(M)

        for i in xrange(self.max_iter):
            # my process
            # h1 = sigmoid(dot(X, w))
            # h1 = h1 * (-y)
            # h2 = X * (-y.reshape((N, 1)))
            # gradient = dot(h1.transpose(), h2)

            # another process
            h = sigmoid(dot(X, w))
            error = (y - h)
            gradient = dot(X.transpose(), error)

            print "Round %d:" % i, gradient

            w = w + self.eta * gradient

        self.w = w
        return w

    def predict(self, X):
        return sign(sigmoid(X * self.w.T) - 0.5)

    def plot(self, X, y):
        N, M = X.shape

        if M != 3:
            print 'Hard to demostrate.'
            return

        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in xrange(n):
            if y[i] == 1:
                xcord1.append(X[i])
                ycord1.append(y[i])
            else:
                xcord2.append(X[i])
                ycord2.append(y[i])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
        ax.scatter(xcord2, ycord2, s = 30, c = 'blue')
        x = arange(-3.0, 3.0, 0.1)
        y = (-self.w[0] - self.w[1]*x)/self.w[2]
        ax.plot(x,y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        return 


def test():
    X, y = load_toy_data()

    my_clf = LogisticRegression()
    print my_clf.train(X, y)

    sk_clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    sk_clf.fit(X, y)
    print sk_clf.coef_

if __name__ == '__main__':
    test()
