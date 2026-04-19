# gradient descent for linear regression by minimizing Mean Squared Error (MSE)

# MSE = (sum(( y_true - y_pred )^2)) / n
# y_pred = (m * x) + b    ==  (our model's prediction)
#  m and b are the parameters we have to optimize
# n : no. of data points

# m = m - (alpha * del(MSE)/del(m))
# b = b - (alpha * del(MSE)/del(b))

import numpy as np
import matplotlib.pyplot as plt

# generate some synthetic data 
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
# true relationship : y = 4 + 3x + noise

# initialize parameters (random values)
m = np.random.randn()
b = np.random.randn()

# hyperparameters 
alpha = 0.1
epochs = 1000
n = len(x)

# gradient descent algorithm
loss_values = []

for i in range(epochs):
    y_pred = (m * x) + b   # predictions
    error = y_pred - y     # difference btw. predicted and true values
    
    # compute gradients
    dm = (2/n) * np.sum(error * x)
    db = (2/n) * np.sum(error)
    
    # update parameters
    m -= alpha * dm
    b -= alpha * db
    
    # compute and store loss
    loss = np.mean(error ** 2)
    loss_values.append(loss)
    
# plot loss over iterations
plt.plot(loss_values)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Descent Convergence')
plt.show()

print(f'Optimised Parameters: m = {m:.4f}, b = {b:.4f}')
    

