######################    Tanh (Hyperbolic Tangent)    ################


#   tanh(x) = ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )
#   range : (-1, 1)
#   better than sigmoid (zero-centered output)
#   demerit : vanishinng gradient


import numpy as np

def tanh(x):
    return np.tanh(x)

x = np.array([-2, 0, 2])
print(tanh(x))
