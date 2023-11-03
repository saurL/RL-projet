from pyboy import PyBoy
from agent import Agent

pyboy = PyBoy('./ROM/Super Mario Bros. Deluxe.gbc')

agent = Agent()
while not pyboy.tick():
    actions = agent.step()
    for action in actions:
        pyboy.send_input(action)
    pyboy.tick() 
    pass
pyboy.stop()