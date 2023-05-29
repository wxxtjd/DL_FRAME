import numpy as np

class Affine:
    def __init__(self, w:np, b:np):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x:np):
        self.x = x
        return np.dot(self.x, self.w) + self.b
    
    def backward(self, diff_grad:np):
        dx = np.dot(diff_grad, self.w.T)
        self.dw = np.dot(self.x.T, diff_grad)
        self.db = np.sum(diff_grad, axis=0)
        return dx
    
class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x:np):
        self.y = 1 / (1+np.exp(-x))
        return self.y
    
    def backward(self, diff_grad):
        dx = diff_grad * self.y * (1 - self.y)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x:np):
        self.mask = (x <= 0)
        x[self.mask] = 0
        return x
    
    def backward(self, diff_grad:np):
        diff_grad[self.mask] = 0
        return diff_grad

class Softmax:
    def __init__(self):
        self.N = None
        self.exp_a = None
        self.sum_exp_a = None

    def forward(self, x:np):
        self.N = x.shape[0]
        maximum = np.max(x, axis=1).reshape(self.N,1)
        self.exp_a = np.exp(x - maximum)
        self.sum_exp_a = np.sum(self.exp_a, axis=1).reshape(self.N,1)
        return self.exp_a / self.sum_exp_a
    
    def backward(self, diff_grad:np):
        reshaped_sum = np.sum(self.exp_a * diff_grad, axis=1).reshape(self.N,1)
        dx = ((reshaped_sum / -self.sum_exp_a**2) + (diff_grad / self.sum_exp_a)) * self.exp_a
        return dx / self.N

class CrossEntropyError:
    def __init__(self):
        self.t = None
        self.y = None
        self.batch_size = None

    def forward(self, y:np, t:np):
        delta = 1e-7
        self.batch_size = y.shape[0]
        self.t, self.y = t, y
        loss_cost = -np.sum(t*np.log(y+delta)) / self.batch_size
        return loss_cost
    
    def backward(self, diff_grad:np):
        dx = -(self.t/self.y)
        return dx