# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.window import Window

# Importing configs
from config import learningCoreSettings, environmentSettings

# Introducing last_x and last_y, used to keep the last point in memory when we draw the tape on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
agentWidth = environmentSettings["agentWidth"]
agentLength = environmentSettings["agentLength"]
sensorSize = environmentSettings["sensorSize"]
sensorSensitivity = environmentSettings["sensorSensitivity"]

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
if(learningCoreSettings["backend"]=="pytorch"):
    from torchDesign import DQN
    brain = DQN(learningCoreSettings)
elif(learningCoreSettings["backend"]=="manual"):
    from manualDesign import DQN
    brain = DQN(learningCoreSettings)
else:
    from UART import Interface
    brain =  Interface()

# interface = Interface()
action2rotation = [0,-1*environmentSettings["rotationDegree"],+1*environmentSettings["rotationDegree"]]
lastReward = 0
Window.clearcolor = (0.96, 0.96, 0.96, 1)
Window.size = (environmentSettings["environmentWidth"], environmentSettings["environmentHeight"])

# Initializing the map
first_update = True
def init():
    global tape
    global first_update
    tape = np.zeros((longueur,largeur))
    first_update = False

# Creating the agent class
class Agent(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    sensor4_x = NumericProperty(0)
    sensor4_y = NumericProperty(0)
    sensor4 = ReferenceListProperty(sensor4_x, sensor4_y)
    sensor5_x = NumericProperty(0)
    sensor5_y = NumericProperty(0)
    sensor5 = ReferenceListProperty(sensor5_x, sensor5_y)
    sensor6_x = NumericProperty(0)
    sensor6_y = NumericProperty(0)
    sensor6 = ReferenceListProperty(sensor6_x, sensor6_y)
    sensor7_x = NumericProperty(0)
    sensor7_y = NumericProperty(0)
    sensor7 = ReferenceListProperty(sensor7_x, sensor7_y)    
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    signal4 = NumericProperty(0)
    signal5 = NumericProperty(0)
    signal6 = NumericProperty(0)
    signal7 = NumericProperty(0)
    Initiated = False

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        if(self.Initiated == False):
            self.pos = Vector(*self.velocity) + (longueur/10,largeur*0.8)
            self.Initiated = True            
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(72, 0).rotate((self.angle-3*environmentSettings["sensorsRotationalDistance"])%360) + self.pos + (agentLength/2-sensorSize/2,agentWidth/2-sensorSize/2)
        self.sensor2 = Vector(72, 0).rotate((self.angle-2*environmentSettings["sensorsRotationalDistance"])%360) + self.pos + (agentLength/2-sensorSize/2,agentWidth/2-sensorSize/2)
        self.sensor3 = Vector(72, 0).rotate((self.angle-1*environmentSettings["sensorsRotationalDistance"])%360) + self.pos + (agentLength/2-sensorSize/2,agentWidth/2-sensorSize/2)
        self.sensor4 = Vector(72, 0).rotate((self.angle)%360) + self.pos + (agentLength/2-sensorSize/2,agentWidth/2-sensorSize/2)
        self.sensor5 = Vector(72, 0).rotate((self.angle+1*environmentSettings["sensorsRotationalDistance"])%360) + self.pos + (agentLength/2-sensorSize/2,agentWidth/2-sensorSize/2)
        self.sensor6 = Vector(72, 0).rotate((self.angle+2*environmentSettings["sensorsRotationalDistance"])%360) + self.pos + (agentLength/2-sensorSize/2,agentWidth/2-sensorSize/2)
        self.sensor7 = Vector(72, 0).rotate((self.angle+3*environmentSettings["sensorsRotationalDistance"])%360) + self.pos + (agentLength/2-sensorSize/2,agentWidth/2-sensorSize/2)

        self.signal1 = int(bool(np.sum(tape[int(self.sensor1_x+sensorSize/2)-sensorSensitivity:int(self.sensor1_x+sensorSize/2)+sensorSensitivity, int(self.sensor1_y+sensorSize/2)-sensorSensitivity:int(self.sensor1_y+sensorSize/2)+sensorSensitivity])))
        self.signal2 = int(bool(np.sum(tape[int(self.sensor2_x+sensorSize/2)-sensorSensitivity:int(self.sensor2_x+sensorSize/2)+sensorSensitivity, int(self.sensor2_y+sensorSize/2)-sensorSensitivity:int(self.sensor2_y+sensorSize/2)+sensorSensitivity])))
        self.signal3 = int(bool(np.sum(tape[int(self.sensor3_x+sensorSize/2)-sensorSensitivity:int(self.sensor3_x+sensorSize/2)+sensorSensitivity, int(self.sensor3_y+sensorSize/2)-sensorSensitivity:int(self.sensor3_y+sensorSize/2)+sensorSensitivity])))
        self.signal4 = int(bool(np.sum(tape[int(self.sensor4_x+sensorSize/2)-sensorSensitivity:int(self.sensor4_x+sensorSize/2)+sensorSensitivity, int(self.sensor4_y+sensorSize/2)-sensorSensitivity:int(self.sensor4_y+sensorSize/2)+sensorSensitivity])))
        self.signal5 = int(bool(np.sum(tape[int(self.sensor5_x+sensorSize/2)-sensorSensitivity:int(self.sensor5_x+sensorSize/2)+sensorSensitivity, int(self.sensor5_y+sensorSize/2)-sensorSensitivity:int(self.sensor5_y+sensorSize/2)+sensorSensitivity])))
        self.signal6 = int(bool(np.sum(tape[int(self.sensor6_x+sensorSize/2)-sensorSensitivity:int(self.sensor6_x+sensorSize/2)+sensorSensitivity, int(self.sensor6_y+sensorSize/2)-sensorSensitivity:int(self.sensor6_y+sensorSize/2)+sensorSensitivity])))
        self.signal7 = int(bool(np.sum(tape[int(self.sensor7_x+sensorSize/2)-sensorSensitivity:int(self.sensor7_x+sensorSize/2)+sensorSensitivity, int(self.sensor7_y+sensorSize/2)-sensorSensitivity:int(self.sensor7_y+sensorSize/2)+sensorSensitivity])))  
   
        if self.pos[0]>longueur-200 or self.pos[0]<10 or self.pos[1]>largeur-10 or self.pos[1]<10:
            self.pos = Vector(*self.velocity) + (longueur/10,largeur*0.8)
            self.angle = 0

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Ball4(Widget):
    pass
class Ball5(Widget):
    pass
class Ball6(Widget):
    pass
class Ball7(Widget):
    pass            

# Creating the game class

class Game(Widget):
    agent = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    ball4 = ObjectProperty(None)
    ball5 = ObjectProperty(None)
    ball6 = ObjectProperty(None)    
    ball7 = ObjectProperty(None)


    # freeze = ObjectProperty(None) 
    freeze = False   


    def serve_agent(self):
        self.agent.center = self.center
        self.agent.velocity = Vector(environmentSettings["agentVelocity"], 0)
    
    def reset(self):
        self.agent.pos = Vector(*self.agent.velocity) + (self.width/10,self.height*0.8)
        self.agent.angle = 0

    def update(self, dt):
        if self.freeze:
            return

        global brain
        global lastReward
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        if(learningCoreSettings["nInputs"]==3):
            lastSignal = [
                self.agent.signal1 or self.agent.signal2,
                self.agent.signal3 or self.agent.signal4 or self.agent.signal5,
                self.agent.signal6 or self.agent.signal7
            ]
        else:
            lastSignal = [
                self.agent.signal1,
                self.agent.signal2,
                self.agent.signal3,
                self.agent.signal4,
                self.agent.signal5,
                self.agent.signal6,
                self.agent.signal7
            ]            

        ballSignal = [
            self.agent.signal1, 
            self.agent.signal2,
            self.agent.signal3,
            self.agent.signal4,
            self.agent.signal5,
            self.agent.signal6,
            self.agent.signal7]

        # Observing the Environment and Fetching the Proper Action    
        action = brain.update(lastReward, lastSignal)

        rotation = action2rotation[action]
        self.agent.move(rotation)
        self.ball1.pos = self.agent.sensor1
        self.ball2.pos = self.agent.sensor2
        self.ball3.pos = self.agent.sensor3
        self.ball4.pos = self.agent.sensor4
        self.ball5.pos = self.agent.sensor5
        self.ball6.pos = self.agent.sensor6
        self.ball7.pos = self.agent.sensor7

        self.balls = [self.ball1,self.ball2,self.ball3,self.ball4,self.ball5,self.ball6,self.ball7]
        self.ball1.color = (1,0,0,1)

        for i in range(len(self.balls)):
            if(ballSignal[i] == 1):
                self.balls[i].color = (1,0,0,1)
            else:
                self.balls[i].color = (0,0,1,1)
        
        self.agent.velocity = Vector(environmentSettings["agentVelocity"], 0).rotate(self.agent.angle)

        if(learningCoreSettings["nInputs"]==3):
            if ((self.agent.signal3 == 1) or (self.agent.signal4 == 1) or (self.agent.signal5 == 1)):
                lastReward = learningCoreSettings["rewardAmount"]
            else: # otherwise
                lastReward = learningCoreSettings["punishAmount"]
        else:
            if (self.agent.signal4 == 1):
                lastReward = learningCoreSettings["rewardAmount"]
            elif (self.agent.signal1 == 0):
                lastReward = 0
            elif (self.agent.signal7 == 0):
                lastReward = 0
            elif (self.agent.signal2 == 0):
                lastReward = 0
            elif (self.agent.signal6 == 0):
                lastReward = 0
            elif (self.agent.signal3 == 0):
                lastReward = 0
            elif (self.agent.signal5 == 0):
                lastReward = 0
            else:
                lastReward = learningCoreSettings["punishAmount"]
     
        if self.agent.x < 10:
            self.agent.x = 10
            lastReward = -1
        if self.agent.x > self.width - 10:
            self.agent.x = self.width - 10
            lastReward = -1
        if self.agent.y < 10:
            self.agent.y = 10
            lastReward = -1
        if self.agent.y > self.height - 10:
            self.agent.y = self.height - 10
            lastReward = -1

# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0,0,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            tape[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            tape[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        self.parent = Game()
        self.parent.serve_agent()
        Clock.schedule_interval(self.parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        rsbtn = Button(text = 'start')
        savebtn = Button(text = 'save', pos = (1*environmentSettings["buttonWidth"],0))
        loadbtn = Button(text = 'load', pos = (2*environmentSettings["buttonWidth"], 0))
        rstbtn = Button(text = 'reset', pos = (3*environmentSettings["buttonWidth"], 0))
        psbtn = Button(text = 'pause', pos = (4*environmentSettings["buttonWidth"], 0))
        clearbtn = Button(text = 'clear', pos = (5*environmentSettings["buttonWidth"], 0))
        monitorbtn = Button(text = 'monitor', pos = (6*environmentSettings["buttonWidth"], 0))

        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        rstbtn.bind(on_release = self.reset)
        psbtn.bind(on_release = self.pause)
        rsbtn.bind(on_release = self.resume)
        monitorbtn.bind(on_release = self.monitor)

        self.parent.add_widget(self.painter)
        self.parent.add_widget(clearbtn)
        self.parent.add_widget(savebtn)
        self.parent.add_widget(loadbtn)
        self.parent.add_widget(rstbtn)
        self.parent.add_widget(psbtn)
        self.parent.add_widget(rsbtn)
        self.parent.add_widget(monitorbtn)
        return self.parent

    def clear_canvas(self, obj):
        global tape
        self.painter.canvas.clear()
        tape = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

    def pause(self, obj):
        print("pausing the simulation...")
        self.parent.freeze=True

    def resume(self, obj):
        print("resuming the simulation...")
        self.parent.freeze=False

    def monitor(self, obj):
        print("monitoring the parameters...")
        self.parent.freeze=True
        brain.monitor()

    def reset(self, obj):
        self.parent.reset()
