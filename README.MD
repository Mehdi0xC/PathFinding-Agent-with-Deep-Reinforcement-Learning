# PathFinding Agent with Deep Reinforcement Learning
This project was done as part of my **B.Sc. Thesis** under the supervision of Dr. Mehdi Sedighi, Computer Engineering Department, Tehran Polytechnic.
The aim was to implement a **reinforcement learning** algorithm to train a pathfinding agent to follow its intended path.
I've defined the project as follows:

1.  The algorithm to be implemented is the famous DQN algorithm, introduced in [this paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).
2.  The implementation should be done in a virtual simulation environment, using the Python language.
---
**Step 1**

The first step was to design a simulation environment. The Kivy framework was used for this purpose. An agent with seven sensors (adjustable) starts in a specific position on a white field. The user can draw lines on the field, and the agent is able to perceive those lines with its sensors.
![Simulation Environment](img/featured.jpg "Simulation Environment")



---
**Step 2**

The PyTorch library is used for the high-level implementation of DQN. For better exploration, the epsilon-greedy method is used.

Hyperparameters were obtained as follows:

1. learning rate = 0.001
2. input layer neurons = 7
3. output layer neurons = 3 (corresponding to three different movements)
4. hidden layer neurons = 10
5. activation function for neurons = ReLU
6. gamma = 0.9
7. reward = 0.1
8. punishment (negative reward) = -1
9. epsilon-greedy iterations = 2500
10. epsilon-greedy initial probability = 0.5

---
**Step 3**

The implementation is done with the NumPy framework using an object-oriented methodology.
![Block Diagram](img/img0.jpg "Block Diagram")


---

**Step 4**

Added a config.py file to have control over the simulation. It would be as follows:

```python
learningCoreSettings = {
    # Number of hidden layer neurons (For three inputs, 8 to 16 neurons work like a charm)
    "nNeurons" : 10,
    # Discount factor (Somewhere between 0.8 and 0.9 is ok)
    "gamma" : 0.9,
    # Replay memory capacity (10000 is more than enough)
    "memoryCapacity" : 10000,
    # Learning rate (Somewhere between 0.0001 to 0.005 is ok) 
    "learningRate" : 0.001,
    # BatchSize, number of samples taken from replay memory in each Learning Iteration
    "batchSize" : 25,
    # Non-linear activation function for neurons, ReLU is used in this project, but you may implement others 
    "activationFunction" :"relu",
    # Number of outputs, can be set to 3 (It's not generic yet)
    "nOutputs" : 3,
    # Number of inputs, can be set to 3 or 7 (It's not generic yet)
    "nInputs" : 3,
    # Regularization factor, for now, it's just implemented in manual design
    "reg" : 0,  
    # AI Backend, can be set to manual, pytorch
    "backend" : "manual",
    # Amount of given reward for the DQN algorithm
    "rewardAmount" : 0.1,
    # Punishment = amount of given negative reward for the DQN algorithm
    "punishAmount" : -1,
    # Softmax temperature, used in the softmax function implementation
    "softmaxTemperature" : 10,
    # Number of iterations in the learning phase
    "learningIterations" : 2500,
    # Number of iterations in the prediction phase
    "predictionIterations" : 2500}


environmentSettings = {    
    "sensorSize" : 15,
    "agentWidth" : 96,
    "agentLength" : 120,
    "rotationDegree" : 3,
    "agentVelocity" : 5,
    "sensorsRotationalDistance":15,
    "sensorSensitivity" : 8,
    "buttonWidth" : 230,
    "environmentWidth" : 800,
    "environmentHeight" : 600
}

**Conclusion**
The robot behaved as expected. We executed it for 5000 iterations: 2500 iterations for the exploration phase (based on the epsilon-greedy method) and another 2500 iterations for the "prediction phase" (where we stopped the learning and fixed the MLP weights). Here, we provided a video showing its functionality.

This is an experimental project suitable for testing and working with small-scale reinforcement learning tasks, which can be used in experimenting with different algorithms for autonomous systems.

