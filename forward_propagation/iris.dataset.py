import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# load iris dataset
iris = datasets.load_iris()
x = iris.data   # features: sepal length, sepal width, petal length, petal width
y = iris.target  # labels: 0, 1, 2 (Setosa, Versicolor, Virginica)

# normalize input features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# define activation functions
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z-np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1,  keepdims=True)

# initialize weights and biases
np.random.seed(42)
n_input = x.shape[1]  # 4 features
n_hidden = 5          # 5 neurons in hidden layer
n_output = 3          # 3 output classes

w1 = np.random.randn(n_input, n_hidden) * 0.01
b1 = np.zeros((1, n_hidden))
w2 = np.random.randn(n_hidden, n_output) * 0.01
b2 = np.zeros((1, n_output))

# forward propagation
z1 = np.dot(x, w1) + b1
a1 = relu(z1)  # hidden layer activations
z2 = np.dot(a1, w2) + b2
a2 = softmax(z2)  # output layer activations

# visualizing activations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Hidden layer activation")
plt.imshow(a1[:50], aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.ylabel("Samples")
plt.xlabel("Classes")

plt.subplot(1, 2, 2)
plt.title("Output layer activation")
plt.imshow(a2[:50], aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.ylabel("Samples")
plt.xlabel("Classes")
plt.show()
