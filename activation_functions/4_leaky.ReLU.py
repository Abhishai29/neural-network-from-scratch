#########################    Leaky ReLU    ################


#   l_ReLU(x) = x, x > 0
#               ax, x <= 0

#   range : (-inf, inf)
#   fixes dead neuron problem
#   commonly used in deep networks
#   a is a small value (e.g. 0.01 )
#   demerit : needs a tuning


import numpy as np

def leaky_relu(x, alpha = 0.01):
    return np.where( x > 0, x, alpha * x )

x = np.array([-2, 0, 2])
print(leaky_relu(x))
