import numpy as np
import matplotlib.pyplot as plt

# w = w - (alpha * dL/dw)
# L : loss fn.
# dL/dw : gradient(slope)
# alpha : learning rate (controls step size) 

# in this e.g:  L = w ** 2
 
# initiate weight
w = 10

# learning rate (step size)
alpha = 0.1

# number of iterations
epochs = 50

# store values for visualization
w_values = []
loss_values = []

for i in range(epochs):
    # compute gradient
    gradient = 2 * w
    
    # update weight
    w = w - alpha * gradient
    
    # store values 
    w_values.append(w)
    loss_values.append(w ** 2)
    
# plot the descent process
plt.plot(loss_values, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Gradient Descent Minimizing L = w^2')
plt.legend()
plt.show()

print('Final weight: ', w)

