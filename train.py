from pyboy import PyBoy
from mario import Mario
from CustomEnv import CustomEnv
from SkipFrame import SkipFrame
from MetricLogger import MetricLogger

import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack


game = "ROM\Super Mario Land (World).gb"
pyboy = PyBoy(game, game_wrapper=True)

env = CustomEnv(pyboy, action_type="press")
env = SkipFrame(env, skip=4)

game_wrapper = pyboy.game_wrapper()
dim = game_wrapper.shape
dir = "checkpoints"
agent = Mario(state_dim=(1, dim[1], dim[0]), action_dim=env.action_space.n, save_dir=dir,)


states_path = 'states'
# File for states!
files = os.listdir(states_path)

episodes = 10000

for i in range(episodes):
    state = env.reset()

    while True:

        # Random action for testing
        action = agent.act(state)


        # Agent performs action
        next_state, reward, done, info = env.step(action)
        agent.Q_learning(state,next_state,action,reward)
        # Update state
        state = next_state

        # Log rewards
        agent.logger.log_step(reward)
        
        # There is a bug however now, if mario gets a mushroo, the game will reset as it will trigger a RAM value that is used for
        # Checking if mario has died

        if done:
            break
    
    agent.logger.log_episode(game_wrapper._level_progress_max)

    if (i % 20 == 0) or (i == episodes - 1):
        # Log information every 20 episode
        agent.record(episode=i)



