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
class lr_model():
    def __init__(self, opt = {}):
        self.alpha = 0.3
        self.maxIter = 20
        self.reg = ''
        try:
            self.alpha = opt['alpha']
        except:
            pass
        try:
            self.maxIter = opt['maxIter']
        except:
            pass
        print 'init a lr model'
        print 'learning rate is %f, maxIter is %d' % (self.alpha, self.maxIter)
    
    def loss(self, y, x, w, reg):
        # x is (n, m) w is (1, m) pre_v is (n, 1)
        pre_v = np.array([[sigmoid(v)] for v in np.dot(x, w.T)])
        error = y - pre_v

        return error
    
    def cal_grad(self, error, x, reg):
        if reg == 'L1':
            pass
        elif reg == 'L2':
            pass
        else:
            return np.dot(error.T, x)

    def update_weight(self, grad, alpha, weight):
        return weight + alpha * grad
    
    def change_alpha_by_loss(self, error, last_error, alpha):
        n, m = error.shape
        if np.sum(abs(error.reshape((n, )))) > np.sum(abs(last_error.reshape((n, )))):
            alpha = alpha * 0.5
        else:
	    	alpha = alpha * 1.05

        return alpha

    def train_model(self, x, y):
        x_t = np.array(x)
        n, m = x_t.shape
        y_t = np.array(y)
        self.error = np.zeros((n, 1))
        self.last_error = np.ones((n, 1))
        self.weight = np.ones((1, m), dtype = 'float32')
        for t in range(self.maxIter):
            self.error = self.loss(y_t, x_t, self.weight, self.reg)
            self.alpha = self.change_alpha_by_loss(self.error, self.last_error, self.alpha)
            self.grad = self.cal_grad(self.error, x_t, self.reg)
            #print 'grad',self.grad
            #print 'learn rate', self.alpha
            self.weight = self.update_weight(self.grad, self.alpha, self.weight)
            self.last_error = self.error
            #print 'weight',self.weight
            #print 'error', sum(abs(self.error))
            #print '===='
        print self.weight

def train_model(train_x, train_y, opt):
    x_t = np.array(train_x)
    n, m = x_t.shape
    y_t = np.array(train_y)
    weight = np.ones((1, m), dtype = 'float32')
    max_iter = opt['maxIter']
    alpha = opt['alpha']
    last_error = np.ones((n, 1))
    for t in range(max_iter):
        pre_v = np.array([[sigmoid(v)] for v in np.dot(x_t, weight.transpose())])
        error = y_t - pre_v
        if np.sum(abs(error.reshape((n,)))) > np.sum(abs(last_error.reshape((n, )))):
            alpha = alpha * 0.5
        else:
	    	alpha = alpha * 1.05
        #print 1.0/ m * sum([e ** 2 for e in error])[0]
       	grad = np.dot(error.T, x_t)
       	weight += grad * alpha
        last_error = error
    print weight
    pre_r = []
    for v in pre_v:
        if v > 0.5:
	        pre_r.append(1)
        else:
            pre_r.append(0)
    #print [int(y[0]) for y in y_t]
    #print pre_r 

opt = {'maxIter': 20, 'alpha': 0.3, 'reg': 'L1'}
x, y = read_data('train_data')
train_model(x, y, opt)
m = lr_model()
m.train_model(x, y)
