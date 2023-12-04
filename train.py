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
episodes = 50000

states_path = 'states'
# File for states!
files = os.listdir(states_path)

agent.loadQ_function()
for i in range(episodes):
    print(f"episode : {i}")
    state = env.reset()

    # 50% chance to start from a checkpoint
    if random.random() > 0.5:
        random_file = random.choice(files)
        file_path = os.path.join(states_path, random_file)
        with open(file_path, 'rb') as file_like_object:
            env.pyboy.load_state(file_like_object)
    while True:

        # Random action for testing
        action = agent.act(state)


        # Agent performs action
        next_state, reward, done, info = env.step(action)
        agent.Q_learning(state,next_state,action,reward)
        # Update state
        state = next_state
        
        # There is a bug however now, if mario gets a mushroo, the game will reset as it will trigger a RAM value that is used for
        # Checking if mario has died
        
        if done or env.respawned == False:
            break
    agent.saveQ_function()

