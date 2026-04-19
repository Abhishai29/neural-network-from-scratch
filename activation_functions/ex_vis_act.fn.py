import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, a = 0.01):
    return np.where( x > 0, x, a * x )

def softmax(x):
    exp_x = np.exp( x - np.max(x) )
    return exp_x / np.sum( exp_x )

y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_leaky_relu = leaky_relu(x)
y_softmax = softmax(x)
y_leaky_relu = np.array(y_leaky_relu, dtype=float).reshape(-1)

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(x, y_relu, label="ReLU", color="r")
plt.title("ReLU")
plt.grid()

plt.subplot(2, 3, 2)
plt.plot(x, y_sigmoid, label="Sigmoid", color="b")
plt.title("Sigmoid")
plt.grid()

plt.subplot(2, 3, 3)
plt.plot(x, y_tanh, label="Tanh", color="purple")
plt.title("Tanh")
plt.grid()

plt.subplot(2, 3, 4)
plt.plot(x, y_leaky_relu, label="leaky_ReLU", color="g")
plt.title("leaky_ReLU")
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(x, y_softmax, label="Softmax", color="orange")
plt.title("Softmax")
plt.grid()

plt.tight_layout()
plt.show()

# print(x.shape, y_leaky_relu.shape)
# print(x.dtype, y_leaky_relu.dtype)