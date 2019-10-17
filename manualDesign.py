import numpy as np
import random 
from myNeuralNetwork import MLP

# Implementing Replay Memory
class ReplayMemory(object):
    # Replay memory object constructor
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # Replay memory object insertion method
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # Replay memory object sampling method
    def sample(self, batchSize):
        samples = random.sample(self.memory, batchSize)
        return samples

# Implementing Deep Q-Learning
class DQN():
    # Implementing Deep Q-Learning constructor
    def __init__(self, settings):
        self.settings = settings
        self.batchSize = settings["batchSize"]
        self.model = MLP(settings)
        self.memory = ReplayMemory(settings["memoryCapacity"])
        self.nInputs = settings["nInputs"]
        self.lastState = np.zeros((1,self.nInputs))
        self.lastAction = 0
        self.lastReward = 0
        self.nOutputs = settings["nOutputs"]
        self.nInputs = settings["nInputs"]
        self.freeze = False


    def monitor(self):
        print("Monitoring the system")
        print("Type the parameters you want to monitor:")
        monitoringInput = input("'model' for model parameters,'data' for memory data\n")
        if monitoringInput=="data":
            for data in self.memory.memory:
                print(data)
        elif monitoringInput=="model":
            print("Model Weights for Layer #1:")
            for w in self.model.W1:
                print (w)
            print("Model Biases for Layer #1:")
            for b in self.model.b1:
                print (b)
            print("Model Weights for Layer #2:")
            for w in self.model.W2:
                print (w)
            print("Model Biases for Layer #2:")
            for b in self.model.b2:
                print (b)
        else:
            print("invalid input")


    # Implementing Deep Q-Learning "call" method
    def update(self, reward, newState):
        if(self.freeze):
            return
        newState = np.array(newState)
        self.memory.push([self.lastState, newState, self.lastAction, reward])
        if (len(self.memory.memory) < self.settings["learningIterations"]):
            action = self.model.predict(newState, True)
        else:
            action = self.model.predict(newState, False)
        if len(self.memory.memory) > self.batchSize and len(self.memory.memory) < self.settings["learningIterations"]:
            samples = self.memory.sample(self.batchSize)
            batchState = np.zeros((self.batchSize,self.nInputs))
            batchNextState = np.zeros((self.batchSize,self.nInputs))
            batchReward = np.zeros((self.batchSize,1))
            batchAction = np.zeros((self.batchSize,1))
            for i in range(len(samples)):
                batchState[i,:] = samples[i][0]
                batchNextState[i,:] = samples[i][1] 
                batchAction[i,:] = samples[i][2] 
                batchReward[i,:] = samples[i][3]
            self.model.learn(batchState, batchNextState, batchReward, batchAction)
        self.lastAction = action
        self.lastState = newState
        return action
