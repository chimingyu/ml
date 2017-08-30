import numpy as np
import math

def read_data(file_name):
    x = []
    y = []
    f = open(file_name, 'r')
    for line in f.readlines():
        fields = line.strip().split('\t')
        x.append([float(fea) for fea in fields[1:-1]])
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
    last_error = 1000
    for t in range(max_iter):
        pre_v = np.array([[sigmoid(v)] for v in np.dot(x_t, weight.transpose())])
        error = y_t - pre_v
        if error[0] > last_error:
	    alpha = alpha * 0.2
	else:
	    alpha = alpha * 1.05
        print 1.0/ m * sum([e ** 2 for e in error])[0]
        grad = np.dot(error.T, x_t)
        weight += grad * alpha
	last_error = error[0]
        #print weight
    pre_r = []
    for v in pre_v:
	if v > 0.5:
	    pre_r.append(1)
	else:
	    pre_r.append(0)
    #print [int(y[0]) for y in y_t]
    #print pre_r 

opt = {'maxIter': 3000, 'alpha': 0.3}
x, y = read_data('train_data')
train_model(x, y, opt)
