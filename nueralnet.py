import numpy as np
from Layers import *
import matplotlib.pyplot as plt

class ANN:
    '''
    Flow Of ANN : Add Layers -> Fit Layers -> Backpropagation(To get gradient)
    '''
    def __init__(self):
        self.layer_sizes = []
        self.act_func = []
        self.layer_params = {}
        self.layer_dict = {}
        self.layer_num = None

    def add_layer(self, size:int, activation_function=None) -> None:
        self.layer_sizes.append(size)
        self.act_func.append(activation_function)

    def fit_layer(self, loss_function:type , weight_init_std=0.01) -> None:
        self.layer_num = len(self.layer_sizes)
        for index in range(1,self.layer_num):
            #Store Weights&Bias in "layer_params" dictionary
            self.layer_params["W"+str(index)] = np.random.rand(self.layer_sizes[index-1], self.layer_sizes[index]) * weight_init_std
            self.layer_params["B"+str(index)] = np.zeros(self.layer_sizes[index])

            #Make NueralNet Frame(Affine layer)
            activation_func = self.act_func[index]
            self.layer_dict["Affine"+str(index)] = Affine(self.layer_params["W"+str(index)], self.layer_params["B"+str(index)])
            self.layer_dict[activation_func.__name__+str(index)] = activation_func()
        #Initialize loss func
        self.loss_func = loss_function()

    def predict(self, x:np) -> np:
        for key in self.layer_dict.keys():
            x = self.layer_dict[key].forward(x)
        return x
    
    def loss(self, y:np, t:np) -> float:
        loss_cost = self.loss_func.forward(y, t)
        return loss_cost
    
    def backpropagation(self, x:np, t:np) -> np:
        #Initialize self variables
        y = self.predict(x)
        loss_cost = self.loss(y, t)

        #Make reversed layer list for backpropagation
        reversed_layer_list = list(self.layer_dict.values())
        reversed_layer_list.reverse()

        #Start point of backpropagation
        diff_grad = 1
        diff_grad = self.loss_func.backward(diff_grad)

        for layer in reversed_layer_list:
            diff_grad = layer.backward(diff_grad)

        grads = {}
        for index in range(1,self.layer_num):
            grads["W"+str(index)] = self.layer_dict["Affine"+str(index)].dw
            grads["B"+str(index)] = self.layer_dict["Affine"+str(index)].db

        return grads, loss_cost
    
    def train(self, x:np, t:np, epochs:int, batch_size:int, optimizer:type, x_test=None, t_test=None, DisplayAcc=False) -> None:
        if DisplayAcc:
            answer = np.argmax(t_test, axis=1)
            test_amount = x_test.shape[0]

        for ep in range(1,epochs+1):
            rand_coor = np.random.choice(x.shape[0],batch_size)
            x_train = x[rand_coor]
            t_train = t[rand_coor]

            grads, loss_cost = self.backpropagation(x_train, t_train)

            for key in grads.keys():
                optimizer.update(self.layer_params[key], grads[key])
            
            if not (ep % 500):
                if DisplayAcc:
                    pred = self.predict(x_test)
                    pred = np.argmax(pred, axis=1)
                    accuracy = float(np.sum(pred==answer) / test_amount)
                    print("Ep : %d  |  Acc : %.3f  |  LOSS : %.5f"%(ep, accuracy, loss_cost))
                else:
                    print("Ep : %d  |  LOSS : %.5f"%(ep, loss_cost))