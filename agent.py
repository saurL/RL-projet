from pyboy import WindowEvent
from numpy import random

class Agent:
    
    def __init__(self,):
        JUMP = [WindowEvent.PRESS_BUTTON_A]
        RIGHT = [WindowEvent.RELEASE_ARROW_LEFT,WindowEvent.PRESS_ARROW_RIGHT]
        LEFT = [WindowEvent.RELEASE_ARROW_RIGHT,WindowEvent.PRESS_ARROW_LEFT]
        STOP_JUMP= [WindowEvent.RELEASE_BUTTON_A]


        self.action = [LEFT,RIGHT,JUMP,STOP_JUMP]

        self.policy=[1/len(self.action) for i in range(len(self.action))]

    def step(self):
        return self.action[random.randint(len(self.action))]