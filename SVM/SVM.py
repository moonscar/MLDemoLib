# coding: utf-8
from numpy import *


def K(X1, X2):
    '''Kernel function.'''
    return dot(X1, X2.transpose())


def g(alpha, b, X, xi, y):
    ''''''
    return dot(K(X, xi).transpose(), alpha*y) + b


def error_measure(alpha, b, X, i, y):
    ''''''
    return g(alpha, b, X, X[i, :].reshape((1, X.shape[1])), y) - y[i] 


def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j


def calcEta(Xi, Xj):
    ''''''
    return K(Xi, Xi) + K(Xj, Xj) - 2* K(Xi, Xj)


def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj


def updateEk(opt, k):#after any alpha has changed update the new value in the cache
    Ek = error_measure(opt.alpha, opt.b, opt.X, k, opt.y)
    opt.eCache[k] = [1,Ek]


def selectJ(i, opt, Ei):
    j = -1
    max_delta_e = 0
    Ej = 0
    valid_e_cache = nonzero(opt.eCache)[0]
    if len(valid_e_cache) > 1:
        for k in valid_e_cache:
            if k!=i:
                Ek = error_measure(opt.alpha, opt.b, opt.X, k, opt.y)
                delta_e = abs(Ei - Ek)
                if delta_e > max_delta_e:
                    max_delta_e = delta_e
                    Ej = Ek
                    j = k
    else:
        j = selectJrand(i, opt.N)
        Ej = error_measure(opt.alpha, opt.b, opt.X, j, opt.y)
    return j, Ej


def inner_loop(i, opt):
    Ei = error_measure(opt.alpha, opt.b, opt.X, i, opt.y)
    if ((opt.y[i]*Ei < -opt.toler) and (opt.alpha[i] < opt.C)) or\
        ((opt.y[i]*Ei > opt.toler) and (opt.alpha[i] > 0)):
        j,Ej = selectJ(i, opt, Ei) 

        alphaIold = opt.alpha[i].copy()
        alphaJold = opt.alpha[j].copy()

        # get range of alpha
        if opt.y[i] != opt.y[j]:
            L = max(0, opt.alpha[j] - opt.alpha[i])
            H = min(opt.C, opt.C + opt.alpha[j] - opt.alpha[i])
        else:
            L = max(0, opt.alpha[j] + opt.alpha[i] - opt.C)
            H = min(opt.C, opt.alpha[j] + opt.alpha[i])
        if L == H:
            print 'L == H'
            return 0

        eta = opt.Kernel[i,i] + opt.Kernel[j,j] - 2.0 * opt.Kernel[i,j]
        if eta <= 0:
            print 'eta <= 0' 
            return 0

        delta_alpha = opt.y[j] * (Ei - Ej)/eta
        opt.alpha[j] = opt.alpha[j] + delta_alpha
        opt.alpha[j] = clipAlpha(opt.alpha[j], H, L)
        updateEk(opt, j)

        if abs(alphaJold - opt.alpha[j]) < 0.00001:
            print 'j not moving enough'
            return 0

        opt.alpha[i] = opt.alpha[i] + opt.y[i]*opt.y[j]*(alphaJold - opt.alpha[j])
        updateEk(opt, i)

        b1 = opt.b - Ei - opt.y[i]*(opt.alpha[i]-alphaIold)*opt.Kernel[i,i]-\
            opt.y[j]*(opt.alpha[j]-alphaJold)*opt.Kernel[i,j]
        b2 = opt.b - Ej - opt.y[i]*(opt.alpha[i]-alphaIold)*opt.Kernel[i,j]-\
            opt.y[j]*(opt.alpha[j]-alphaJold)*opt.Kernel[j,j]

        if (0 < opt.alpha[i]) and (opt.alpha[i] < opt.C):
            opt.b = b1
        elif (0 < opt.alpha[j]) and (opt.alpha[j] < opt.C):
            opt.b = b2
        else:
            opt.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.N = dataMatIn.shape[0]
        self.y = classLabels.reshape((self.N, 1))
        self.C = C
        self.toler = toler
        self.alpha = zeros((self.N,1))
        self.b = 0
        self.eCache = zeros((self.N,2)) #first column is valid flag
        self.Kernel = zeros((self.N,self.N))
        for i in range(self.N):
            t = K(self.X, self.X[i,:].reshape((1, dataMatIn.shape[1])))
            self.Kernel[:,i] = t[:, 0]



class SVM(object):
    """docstring for SVM"""
    def __init__(self, C = 0.6, toler = 0.001, maxIter = 40):
        self.C = C
        self.toler = toler
        self.maxIter = maxIter

    def train(self, X, y):
        opt = optStruct(X, y, self.C, self.toler)
        iteration = 0
        entire_set = True
        alpha_pairs_changed = 0
        while (iteration < self.maxIter) and (alpha_pairs_changed > 0 or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for i in xrange(opt.N):
                    alpha_pairs_changed += inner_loop(i, opt)
                iteration += 1
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iteration,i,alpha_pairs_changed)
            else:
                non_bound = nonzero((opt.alpha > 0) * (opt.alpha < opt.C))[0]
                for i in non_bound:
                    alpha_pairs_changed += inner_loop(i, opt)
                iteration += 1
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iteration,i,alpha_pairs_changed)
            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True

        self.b = opt.b
        sv = nonzero(opt.alpha)[0]
        self.sv_x = opt.X[sv]
        self.sv_y = opt.y[sv]
        self.sv_alpha = opt.alpha[sv]
        return opt.alpha, opt.b

    def simTrain(self, X, y, C, toler, maxIter):
        N, M = X.shape
        y = y.reshape((N, 1))
        alpha = zeros((N, 1))
        old_iter = 0
        iteration = 0
        b = 0
        while iteration < maxIter:
            alpha_pairs_changed = 0
            for i in xrange(N):     # 外部alpha选择机制
                Ei = error_measure(alpha, b, X, i, y)
                if ((y[i]*Ei < -toler) and (alpha[i] < C)) or\
                    ((y[i]*Ei > toler) and (alpha[i] > 0)):
                    j = selectJrand(i, N) # 内部alpha选择机制
                    Ej = error_measure(alpha, b, X, j, y)
                    # bak alpha[i] and alpha[j]
                    alphaIold = alpha[i].copy(); alphaJold = alpha[j].copy();

                    # get range of alpha
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(C, C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[j] + alpha[i] - C)
                        H = min(C, alpha[j] + alpha[i])
                    if L == H:
                        print 'L == H'
                        continue

                    eta = calcEta(X[i].reshape((1, M)), X[j].reshape((1, M)))
                    if eta <= 0:
                        print 'eta <= 0' 
                        continue

                    delta_alpha = y[j] * (Ei - Ej)/eta
                    alpha[j] = alpha[j] + delta_alpha
                    alpha[j] = clipAlpha(alpha[j], H, L)

                    if abs(alphaJold - alpha[j]) < 0.00001:
                        print 'j not moving enough'
                        continue

                    alpha[i] = alpha[i] + y[i]*y[j]*(alphaJold - alpha[j])

                    b1 = b - Ei - y[i]*(alpha[i]-alphaIold)*K(X[i].reshape(1, M), X[i].reshape(1,M))-\
                    y[j]*(alpha[j]-alphaJold)*K(X[i].reshape(1, M), X[j].reshape(1,M))
                    b2 = b - Ej - y[i]*(alpha[i]-alphaIold)*K(X[i].reshape(1, M), X[j].reshape(1,M))-\
                    y[j]*(alpha[j]-alphaJold)*K(X[j].reshape(1, M), X[j].reshape(1,M))

                    if (0 < alpha[i]) and (alpha[i] < C):
                        b = b1
                    elif (0 < alpha[j]) and (alpha[j] < C):
                        b = b2
                    else:
                        b = (b1 + b2)/2.0
                    alpha_pairs_changed += 1
                    print "iter: %d i:%d, pairs changed %d" % (iteration,i,alpha_pairs_changed)
            if alpha_pairs_changed == 0:
                iteration += 1
            else:
                iteration = 0
            if old_iter != iteration:
                print "iteration number: %d" % iteration
                old_iter = iteration
        return b, alpha

    def predict(self, X):
        return g(self.sv_alpha, self.b, self.sv_x, X, self.sv_y)

    def plot(self):
        return 


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return array(dataMat),array(labelMat)

def test():
    a = SVM()
    X, y = loadDataSet()
    y = y.reshape((X.shape[0], 1))
    a, b = a.train(X, y)
    print b, a[a!=0]

def test2():
    a = SVM()
    X, y = loadDataSet()
    y = y.reshape((X.shape[0], 1))
    b, a = a.simTrain(X, y, 0.6, 0.001, 40)
    print b, a[a!=0]


if __name__ == '__main__':
    test()