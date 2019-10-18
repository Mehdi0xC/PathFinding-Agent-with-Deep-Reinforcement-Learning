learningCoreSettings = {
    # Number of hidden layer neurons (For three inputs, 8 to 16 neurons work like charm)
    "nNeurons" : 10,
    # Discount factor (Somewhere between 0.8 and 0.9 is ok)
    "gamma" : 0.9,
    # Replay memory capacity (10000 is more than enough)
    "memoryCapacity" : 10000,
    # Learning-rate (Somewhere between 0.0001 to 0.005 is ok) 
    "learningRate" : 0.001,
    # BatchSize, number of samples taken from replay memory in each Learning Iteration
    "batchSize" : 25,
    # Non-linear activation function for neurons, ReLU is used in this project but you may implement others 
    "activationFunction" :"relu",
    # Number of outputs, can be set to 3 (Its not generic yet)
    "nOutputs" : 3,
    # Number of inputs, can be set to 3 or 7 (Its not generic yet)
    "nInputs" : 3,
    # Regularization factor, for now its just implemented in manual design
    "reg" : 0,  
    # AI Backend, can be set to manual, pytorch or UART
    "backend" : "manual",
    # Amount of given reward for DQN algorithm
    "rewardAmount" : 0.1,
    # Punishment = amount of given negative reward for DQN algorithm
    "punishAmount" : -1,
    # Softmax temperature, used in softmax function implementation
    "softmaxTemperature" : 10,
    # Number of iterations in learning phase
    "learningIterations" : 2500,
    # Number of iterations in prediction phase
    "predictionIterations" : 2500}

learningDeviceSettings = {
        # Device port name, if using UART as backend
        "devicePortName" : "COM5",
        # UART connection baudrate
        "baudrate" : 115200
    }

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
