import numpy as np

x = np.random.uniform(-5, 5, 10)
print("Input:\n, x")

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

print("\nReLU\n", relu(x))
print("\nSigmoid\n", sigmoid(x))
print("\nTanh\n", tanh(x))
print("\nleaky_ReLU\n", leaky_relu(x))
print("\nSoftmax\n", softmax(x))



