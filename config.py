core_settings = {
    # Number of hidden layer neurons (For three inputs, 8 to 16 neurons work like charm)
    "n_neurons": 10,
    # Discount factor (Somewhere between 0.8 and 0.9 is ok)
    "gamma": 0.9,
    # Replay memory capacity (10000 is more than enough)
    "memory_capacity": 10000,
    # Learning-rate (Somewhere between 0.0001 to 0.005 is ok) 
    "learning_rate": 0.001,
    # Batch size, number of samples taken from replay memory in each learning iteration
    "batch_size": 25,
    # Non-linear activation function for neurons, ReLU is used in this project but you may implement others 
    "activation_function": "relu",
    # Number of outputs, can be set to 3 (It's not generic yet)
    "n_outputs": 3,
    # Number of inputs, can be set to 3 or 7 (It's not generic yet)
    "n_inputs": 3,
    # Regularization factor, for now it's just implemented in manual design
    "reg": 0,  
    # AI backend, can be set to manual, pytorch or UART
    "backend": "manual",
    # Amount of given reward for DQN algorithm
    "reward_amount": 0.1,
    # Punishment = amount of given negative reward for DQN algorithm
    "punish_amount": -1,
    # Softmax temperature, used in softmax function implementation
    "softmax_temperature": 10,
    # Number of iterations in learning phase
    "learning_iterations": 2500,
    # Number of iterations in prediction phase
    "prediction_iterations": 2500
}

environment_settings = {
    # Size of the robot sensors 
    "sensor_size": 15,
    # Width of the robot
    "agent_width": 96,
    # Length of the robot
    "agent_length": 120,
    # Degree of rotation for each rotation action
    "rotation_degree": 3,
    # Velocity of the robot in the environment
    "agent_velocity": 5,
    # Degree between each sensor
    "sensors_rotational_distance": 15,
    # Sensor sensitivity
    "sensor_sensitivity": 8,
    # Width of the button in the simulation
    "button_width": 230,
    # Width of the simulation environment
    "environment_width": 800,
    # Height of the simulation environment
    "environment_height": 600
}
