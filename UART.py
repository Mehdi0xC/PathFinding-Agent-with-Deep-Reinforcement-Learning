import serial
from config import learningDeviceSettings

# Interface Initialization
class Interface(object):
    def __init__(self):
        self.port = serial.Serial(
        port=learningDeviceSettings["devicePortName"],
        baudrate=learningDeviceSettings["baudrate"],
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
            timeout=None)

# Interfacing Method 
    def update(self,reward, newState): 
        command = ""
        for s in newState:
            if(s):
                command = command + "1"
            else:
                command = command + "0"
        self.port.write((command + '\n').encode('utf-8'))
        command = str(reward)
        self.port.write((command + '\n').encode('utf-8'))    
        action = self.port.readline().decode('utf-8')
        return int(action)
