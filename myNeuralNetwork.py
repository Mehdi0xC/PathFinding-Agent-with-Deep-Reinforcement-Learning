import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp
from utils import softmax, relu
from sklearn.utils import shuffle

class MLP(object):
    def __init__(self,settings):
        self.gamma = settings["gamma"]
        self.nNeurons = settings["nNeurons"]
        self.learningRate = settings["learningRate"]
        self.batchSize = settings["batchSize"]
        self.activationFunction = settings["activationFunction"]
        self.nOutputs = settings["nOutputs"]
        self.nInputs = settings["nInputs"]
        #input to hidden layer weights and biases
        self.W1 = np.random.randn(self.nInputs, self.nNeurons) / np.sqrt(self.nInputs + self.nNeurons)
        self.b1 = 0.01+np.zeros(self.nNeurons)
        #hidden layer to output weights and biases
        self.W2 = np.random.randn(self.nNeurons, self.nOutputs) / np.sqrt(self.nNeurons + self.nOutputs)
        self.b2 = 0.01+np.zeros(self.nOutputs) 
        self.reg = settings["reg"]

    def learn(self, batchState, batchNextState, batchReward, batchAction):
        hiddenLayerOutput = relu(batchState.dot(self.W1) + self.b1)
        output = hiddenLayerOutput.dot(self.W2) + self.b2
        target = cp(output)
        hiddenLayerOutput2 = relu(batchNextState.dot(self.W1) + self.b1)
        nextOutputs = hiddenLayerOutput2.dot(self.W2) + self.b2 
        maxIndices = np.argmax(nextOutputs,axis=1)
        for i in range(self.batchSize):
            target[i,batchAction[i].astype(int)] =  self.gamma*nextOutputs[i,maxIndices[i]] + batchReward[i]
        # gradient descent step
        distance = target - output
        self.W2 += self.learningRate*(hiddenLayerOutput.T.dot(distance) + self.reg*self.W2)
        self.b2 += self.learningRate*(distance.sum(axis=0) + self.reg*self.b2)
        dOutput = distance.dot(self.W2.T) * (hiddenLayerOutput > 0) # relu
        self.W1 += self.learningRate*(batchState.T.dot(dOutput) + self.reg*self.W1)
        self.b1 += self.learningRate*(dOutput.sum(axis=0) + self.reg*self.b1)

    def predict(self, X, learning):
        hiddenLayerOutput = relu(X.dot(self.W1) + self.b1)
        output = hiddenLayerOutput.dot(self.W2) + self.b2
        if(learning):
            probs = softmax(output)
            action = np.where(probs == np.random.choice(probs, 1,p=probs))
            return (int(action[0]))
        return np.argmax(output)
