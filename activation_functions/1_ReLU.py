############################    ReLU    #############################

#   ReLU(x) = max(0, x)
#   range = [0, inf)
#   used in : most deep learning models
#   demerit : dead neurons (zero gradient for -ve inputs)


import numpy as np

def relu(x):
    return np.maximum(0, x)

x = np.array([-2, 0 , 2])
print(relu(x))
