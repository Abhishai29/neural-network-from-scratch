import numpy as np

# sample == row , feature == column

#input features (2 samples, 3 features each)
x = np.array([[0.5, 0.2, 0.1], [0.9, 0.8, 0.5]])

# layers : input --> hidden --> output
# weights for input to hidden layer (3 input features -> 4 neurons)
w1 = np.random.randn(3, 4)  # shape (3,4)
b1 = np.zeros((1, 4))       # bias shape (1, 4)

# weights for hidden to output layer (4 neurons -> 2 output features)
w2 = np.random.randn(4, 2)  # shape (4, 2)
b2 = np.zeros((1, 2))       # bias shape (1, 2)

# hidden layer computation: z1 = x.w1 + b1
z1 = np.dot(x, w1) + b1

# apply activation fn. (ReLU)
a1 = np.maximum(0, z1)  # ReLU activation

# output layer computation: z2 = a1.w2 + b2
z2 = np.dot(a1, w2) + b2 

# apply activation fn. (softmax for classification)
exp_values = np.exp(z2 - np.max(z2, axis=1, keepdims=True))  # stability trick
a2 = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # softmax activation

# output
print(a2)
