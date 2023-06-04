import numpy as np
from Layers import *
import pickle
import os
#Dictionary Keys for Save
SAVE_KEYS = ["layer_sizes", "layer_params", "layer_dict", "loss_func"]

def model_load(filename:str, dir="models/"):
    path = dir + filename + ".pkl"
    with open(path, "rb") as f:
        savepoint = pickle.load(f)
    model = ANN()
    model.load(savepoint[SAVE_KEYS[0]], savepoint[SAVE_KEYS[1]],
               savepoint[SAVE_KEYS[2]], savepoint[SAVE_KEYS[3]])
    return model

class ANN:
    '''
    Recommended Flow Of ANN : Add Layers -> Fit Layers -> Backpropagation(To get gradient)
    '''
    def __init__(self):
        self.layer_sizes = []
        self.act_func = []
        self.layer_params = {}
        self.layer_dict = {}
        self.layer_num = None
        self.loss_func = None

    def load(self, layer_sizes:list, layer_params:dict, layer_dict:dict, loss_func:type):
        self.layer_sizes = layer_sizes
        self.layer_params = layer_params
        self.layer_dict = layer_dict
        self.loss_func = loss_func
        self.layer_num = len(layer_sizes)

    def save(self, filename:str, dir="models/"):
        try:
            #initialize filename & path
            filename = filename + ".pkl"
            path = dir + filename

            #Check decision of user for duplicate name
            if filename in os.listdir(dir):
                decision = input("Duplicate file name. Would you like to overwrite it? [Y/N] ")
            if decision.lower() == 'n':
                raise Exception

            #Make dictionary of ANN elements for save
            elements = {SAVE_KEYS[0]: self.layer_sizes, SAVE_KEYS[1]: self.layer_params,\
                        SAVE_KEYS[2]: self.layer_dict, SAVE_KEYS[3]: self.loss_func}
            
            #save
            with open(path, "wb") as f:
                pickle.dump(elements, f)
        
        except Exception as e:
            print("[ERROR] The model didn't save completely.")
            print(e)

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
    
    def train(self, x:np, t:np, epochs:int, batch_size:int, optimizer:type, x_test=None, t_test=None, interval=100, DisplayAcc=False) -> None:
        if DisplayAcc:
            answer = np.argmax(t_test, axis=1)
            test_amount = x_test.shape[0]

        for ep in range(1,epochs+1):
            rand_coor = np.random.choice(x.shape[0],batch_size)
            x_train = x[rand_coor]
            t_train = t[rand_coor]

            grads, loss_cost = self.backpropagation(x_train, t_train)

            optimizer.update(self.layer_params, grads)

            if not (ep%interval):
                if DisplayAcc:
                    pred = self.predict(x_test)
                    pred = np.argmax(pred, axis=1)
                    accuracy = float(np.sum(pred==answer) / test_amount)
                    print("Ep : %d  |  Acc : %.3f  |  LOSS : %.5f"%(ep, accuracy, loss_cost))
                else:
                    print("Ep : %d  |  LOSS : %.5f"%(ep, loss_cost))