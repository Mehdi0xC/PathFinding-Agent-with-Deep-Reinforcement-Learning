learningCoreSettings = {
    "nNeurons" : 10,
    "gamma" : 0.9,
    "memoryCapacity" : 10000,
    "learningRate" : 0.001,
    "batchSize" : 25,
    "activationFunction" :"relu",
    "nOutputs" : 3,
    "nInputs" : 3,
    "reg" : 0,  
    "backend" : "pytorch",
    "rewardAmount" : 0.1,
    "punishAmount" : -1,
    "softmaxTemperature" : 10,
    "learningIterations" : 1000,
    "predictionIterations" : 1000,
    "device" : "COM5"}

learningDeviceSettings = {
        "devicePortName" : "COM5",
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
    "buttonWidth" : 230
}
