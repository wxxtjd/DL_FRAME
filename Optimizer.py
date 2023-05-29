import numpy as np

class SGD:
    def __init__(self, lr=0.01) :
        self.lr = lr

    def update(self, x:np, dx:np) -> np:
        for key in x.keys():
            x[key] -= self.lr * dx[key]
            # return x

class Momentum:
    def __init__(self, lr=0.01, m=0.9):
        self.v = {}
        self.lr = lr
        self.m = m

    def update(self, x:np, dx:np):
        if not len(self.v):
            for key, value in x.items():
                self.v[key] = np.zeros_like(value)
        
        for key in x.keys():
            self.v[key] = self.m * self.v[key] - self.lr * dx[key]
            x[key] += self.v[key]