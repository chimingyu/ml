import numpy as np
import math

def read_data(file_name):
    x = []
    y = []
    f = open(file_name, 'r')
    for line in f.readlines():
        fields = line.strip().split('\t')
        x.append([float(fea) for fea in fields[1:-2]])
        y.append([float(fields[-1])])
    f.close()
    return x, y

def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))

def train_model(train_x, train_y, opt):
    x_t = np.array(train_x)
    n, m = x_t.shape
    y_t = np.array(train_y)
    weight = np.ones((1, m), dtype = 'float32')
    max_iter = opt['maxIter']
    alpha = opt['alpha']
    for t in range(max_iter):
        pre_v = np.array([[sigmoid(v)] for v in np.dot(x_t, weight.transpose())])
        error = y_t - pre_v
        #print 1.0/ m * sum([e ** 2 for e in error])
        grad = np.dot(error.T, x_t)
        weight += grad * alpha
        #print weight
    pre_r = []
    for v in pre_v:
	if v > 0.5:
	    pre_r.append(1)
	else:
	    pre_r.append(0)
    print [int(y[0]) for y in y_t]
    print pre_r 

opt = {'maxIter': 10, 'alpha': 0.3}
x, y = read_data('train_data')
train_model(x, y, opt)
