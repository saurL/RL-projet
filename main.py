from pyboy import PyBoy
from mario import Mario
from CustomEnv import CustomEnv
from SkipFrame import SkipFrame

import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

dir = "placeholder"
game = "ROM\Super Mario Land (World).gb"
pyboy = PyBoy(game, game_wrapper=True)

env = CustomEnv(pyboy, action_type="press")
env = SkipFrame(env, skip=4)

game_wrapper = pyboy.game_wrapper()
dim = game_wrapper.shape

agent = Mario(state_dim=(1, dim[1], dim[0]), action_dim=env.action_space.n, save_dir=dir)
agent.exploration_rate = 1
episodes = 5000000
start = True

if (not agent.loadQ_function()):
    if(not agent.loadQ_function()):
        print("the file could not be oppenned")
        start= False

if start:
    agent.checkDict()
    for i in range(episodes):
        print(f"episode : {i}")
        state = env.reset()
        while True:

            # Random action for testing
            action = agent.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)
            agent.Q_learning(state,next_state,action,reward)
            print(reward)
            # Update state
            state = next_state

            if done:
                break
        agent.saveQ_function()

