import numpy as np
from Layers import *
from nueralnet import *
from Optimizer import *
from dataset.mnist import load_mnist

#Get Data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#Initialize Detail Options of Model
optimizer = Minibatch(lr=0.3)
batch_size = 100
epochs = 10000
test_data_size = 5000

#Make Model
model = ANN()

#Form Layers of Model
model.add_layer(784)#input layer
model.add_layer(size=50, activation_function=Relu)#hidden layer1
model.add_layer(size=10, activation_function=Softmax)#output layer
model.fit_layer(CrossEntropyError)

#Train Model
model.train(x=x_train, t=t_train, epochs=epochs, batch_size=batch_size, optimizer=optimizer,
            x_test=x_test[:test_data_size], t_test=t_test[:test_data_size], DisplayAcc=True)
