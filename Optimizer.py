import numpy as np

class Minibatch:
    def __init__(self, lr=0.01) :
        self.lr = lr

    def update(self, x:np, dx:np) -> np:
        x -= self.lr * dx
        # return x