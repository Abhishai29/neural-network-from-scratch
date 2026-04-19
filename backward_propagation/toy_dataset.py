import numpy as np

# Toy dataset (2 samples, 2 features)

x = np.array([[0,0],[1,1]])   # inputs
y = np.array([[0],[1]])       # targets (binary classification)

# initialize weights and biases randomly
np.random.seed(42)
w1 = np.random.randn(2,2)   # weights from input to hidden
b1 = np.zeros((1,2))        # bias for hidden layer

w2 = np.random.randn(2,1)   # weights from hidden to output
b2 = np.zeros((1,1))        # bias for output layer

# sigmoid fn.
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )

# forward prop
z1 = np.dot(x, w1) + b1     # hidden layer
a1 = sigmoid(z1)

z2 = np.dot(a1, w2) + b2    # output layer
a2 = sigmoid(z2)

print("z1:", z1)
print("a1:", a1)
print("z2:", z2)
print("a2(output):", a2)

# binary cross-entropy loss :  -1/n * sum[y(log(a2))+(1-y)log(1-a2)]
m = y.shape[0]  # number of samples
loss = - np.mean(y * np.log(a2 + 1e-8) + (1-y) * np.log(1-a2 + 1e-8))
print("Loss:", loss)

# backward pass
dz2 = a2 - y                    # derivative of loss w.r.t z2
dw2 = np.dot(a1.T, dz2) / m     # shape: (1,2)
db2 = np.sum(dz2, axis=0, keepdims=True) / m    # shape: (1,1)

da1 = np.dot(dz2, w2.T)         # shape: (2,2)
dz1 = da1 * a1 * (1 - a1)       # derivative of sigmoid


