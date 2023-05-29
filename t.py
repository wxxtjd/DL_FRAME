import numpy as np

dict_ = {}
layer = {}
for i in range(1,3):
    dict_["W"+str(i)] = np.array([1,2,3])
    dict_["B"+str(i)] = np.zeros(1)

    layer["A"+str(i)] = dict_["W"+str(i)]


for i in range(1,3):
    print(id(dict_["W"+str(i)]), id(layer["A"+str(i)]))