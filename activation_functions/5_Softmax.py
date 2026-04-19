###########################    Softmax    ####################


#   softmax(xi) = exp(xi) / sum(exp(xi))
#   range : (0, 1)
#   used in : muti-class classification   #
#   converts raw scores into probabilities
#   values sum to 1
#   demerit : computationally expensive


import numpy as np

def softmax(x):
    exp_x = np.exp( x - np.max(x) )    # subtract max(x) for numerical stability
    return exp_x / np.sum(exp_x)

x = np.array([-2, 0, 2])
print(softmax(x))