from itertools import *
from numpy import *

def gini(prob):
    return 1 - sum([p**2 for p in prob])


def preprocess_for_continuous(features):
    dist = {}
    sequences = sorted(features)
    for i, x in enumerate(features):
        dist[x] = i

    for k in dist.keys():
        dist[k] = dist[k]/float(len(features))

    return dist


def gini_gain_for_cont(pivot, X):
    return (len(T1)*gini(T1)+len(T2)*gini(T2))/(len(T1)+len(T2))


def choose_pivot_for_continuous(X):
    key = set(sorted(X))
    max_gini_gain = 0
    best_pivot = None
    for i in xrange(len(dist.keys())-1):
        pivot = (k[i] + k[i+1])/2.0
        gg = gini_gain_for_cont(pivot, D)
        if gg > max_gini_gain:
            max_gini_gain = gg
            best_pivot = pivot
    return best_pivot


def preprocess_for_discrete(features):
    '''Generate a frequence distribute for data.'''
    dist = {}
    for e in features:
        if dist.has_key(e):
            dist[e] += 1
        else:
            dist[e] = 1.0

    for k in dist.keys():
        dist[k] = dist[k]/len(features)

    return dist


def gini_gain_for_disc(subset1, subset2, X):
    '''Caculate gini_gain for a divided of data.'''
    data_len = len(X)
    dist = preprocess_for_discrete(X)
    fraction = 0.0
    arr = []
    for s in subset1:
        fraction += dist[s]
        arr.append(dist[s])

    gini_gain = fraction * gini(arr)
    fraction = 0.0
    arr = []

    for s in subset1:
        fraction += dist[s]
        arr.append(dist[s])

    gini_gain += fraction * gini(arr)
    return gini_gain


def choose_class_for_discrete(X):
    '''Choose sub class for discrete data.'''
    data = list(X)
    enum = set(data)
    max_gini_gain = 0
    best_set = None
    # Two levels of iteration is just for get a subset. 
    for i in xrange(1, (len(enum)+1)/2):
        for s in combinations(enum, i):
            gg = gini_gain_for_disc(s, enum.symmetric_difference(s), data)
            if gg > max_gini_gain:
                max_gini_gain = gg
                best_set = s
    return max_gini_gain, best_set


# 进度：
# 基本可以完成建造决策树（对离散特征），递归两层以上会出问题待修复
# 节点可以使用
# 无法判断离散/连续特征
# 无法对连续特征判断并建立决策树
# 预剪枝功能未完善
# 后剪枝功能未实现
# 预测功能未实现
# 决策树可视化功能未实现
class cart_tree(object):
    """docstring for cart_tree"""
    def __init__(self):
        return

    def continuous_or_discrete(self, data):
        '''Something should be instead of.'''
        return False

    def abort_creating(self, data):
        '''When some conditions is satisfy, 
        abort creating tree. Pre-pruning step.'''
        if len(data) < 10:
            return True
        else:
            return False

    def create_tree(self, X):
        '''Something should be instand of.'''
        # data = hstack((X, y))
        data = X
        main_tree = {}

        # pre-prune
        if self.abort_creating(data):
            return main_tree

        best_gini_gain = 0.0
        seperation_dim = -1
        features_value = None
        c_d = None

        data_size, dimension = data.shape

        for d in xrange(dimension):
            continuous = self.continuous_or_discrete(data[:, d])
            cur_gini_gain = 0.0
            f_value = None
            if continuous:
                cur_gini_gain, f_value = choose_pivot_for_continuous(data[:, d])
            else:
                cur_gini_gain, f_value = choose_class_for_discrete(data[:, d])

            if cur_gini_gain > best_gini_gain:
                best_gini_gain = cur_gini_gain
                seperation_dim = d
                features_value = f_value
                c_d = continuous

        # seperate the data
        if c_d:
            X1 = data[data[:, seperation_dim] <= features_value, :]
            X2 = data[data[:, seperation_dim] <= features_value, :]
        else:
            X1 = array(filter(lambda x: x[seperation_dim] in features_value, data))
            X2 = array(filter(lambda x: x[seperation_dim] not in features_value, data))

        main_tree['info'] = (seperation_dim, c_d, features_value)
        main_tree['left'] = self.create_tree(X1)
        main_tree['right'] = self.create_tree(X2)

        return main_tree

    def prune(self):
        '''A method for post-prune.'''

    def tree_plot(self):
        '''A visualization of this tree.'''

    def predict(self, X):
        return None 

if __name__ == '__main__':
    main()
