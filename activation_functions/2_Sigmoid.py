#####################    SIGMOID FUNCTION    ###############

#   sig(x) = 1/(1 + exp(-x))
#   range : (0, 1)
#   used in : binary classification
#   demerit : vanishing gradient (small derivative for small/large inputs)


import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

x = np.array([-2, 0 , 2]) 
print(sigmoid(x))
