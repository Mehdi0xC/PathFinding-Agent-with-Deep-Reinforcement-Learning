# Importing Libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the Architecture of the Neural Network
class Network(nn.Module):
    def __init__(self, nInputs, nOutput, nNeuron):
        super(Network, self).__init__()
        self.nInputs = nInputs
        self.nOutput = nOutput
        self.fc1 = nn.Linear(nInputs, nNeuron)
        self.fc2 = nn.Linear(nNeuron, nOutput)

# Feedforward Propagation
    def forward(self, state):
        x = F.relu(self.fc1(state))
        qValues = self.fc2(x)
        return qValues

# Implementing Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

# Pushing Data to Replay Memory    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

# Sampling Replay Memory  
    def sample(self, batchSize):
        samples = zip(*random.sample(self.memory, batchSize))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q-Learning
class DQN():
    def __init__(self, settings):
        self.settings = settings
        self.gamma = settings["gamma"]
        self.rewardWindow = []
        self.model = Network(settings["nInputs"], settings["nOutputs"],settings["nNeurons"])
        self.memory = ReplayMemory(settings["memoryCapacity"])
        self.optimizer = optim.Adam(self.model.parameters(), lr = settings["learningRate"])
        self.lastState = torch.Tensor(settings["nInputs"]).unsqueeze(0)
        self.lastAction = 0
        self.lastReward = 0
    
# Implementing DQN Policy  
    def selectAction(self, state):
        if(len(self.memory.memory) < self.settings["learningIterations"]):
            with torch.no_grad():
                probs = F.softmax(self.model(Variable(state))*self.settings["softmaxTemperature"], dim=0)
        else:            
            with torch.no_grad():
                action = np.argmax(self.model(Variable(state)).numpy(),1)
                return action[0]
        action = probs.multinomial(1)
        return int(action.data[0,0])

# Implementing DQN Learn Function      
    def learn(self, batchState, batchNextState, batchReward, batchAction):
        outputs = self.model(batchState).gather(1, batchAction.unsqueeze(1)).squeeze(1)
        nextOutputs = self.model(batchNextState).detach().max(1)[0]
        target = self.gamma*nextOutputs + batchReward
        TDLoss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        TDLoss.backward()
        self.optimizer.step()

# Implementing DQN Update Function (Integrating Learn and Policy)
    def update(self, reward, newSignal):
        newState = torch.Tensor(newSignal).float().unsqueeze(0)
        self.memory.push((self.lastState, newState, torch.LongTensor([int(self.lastAction)]), torch.Tensor([self.lastReward])))
        action = self.selectAction(newState)
        if len(self.memory.memory) > self.settings["batchSize"]:
            batchState, batchNextState, batchAction, batchReward = self.memory.sample(self.settings["batchSize"])
            self.learn(batchState, batchNextState, batchReward, batchAction)
        self.lastAction = action
        self.lastState = newState
        self.lastReward = reward
        return action
    
# Score Function to Evaluate the Algorithm
    def score(self):
        return sum(self.rewardWindow)/(len(self.rewardWindow)+1.)

# Function to Save the Model  
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')

# Function to Load the Model    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")