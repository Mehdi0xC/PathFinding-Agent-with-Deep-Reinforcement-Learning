import numpy as np
from config import learningCoreSettings

# Weight initialization function for MLP
def initWeights(inputSize, outputSize):
    W = np.random.randn(inputSize, outputSize) / np.sqrt(inputSize)
    b = np.zeros(outputSize)
    return W.astype(np.float32), b.astype(np.float32)

# Relu activation function
def relu(x):
    return x * (x > 0)

# Sigmoid activation function
def sigmoid(A):
    return 1 / (1 + np.exp(-A))

# Softmax layer implementation
def softmax(A):
    expA = np.exp(A/learningCoreSettings["softmaxTemperature"])
    return expA / expA.sum()
