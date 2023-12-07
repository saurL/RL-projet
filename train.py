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
agent = Mario(state_dim=(1, dim[1], dim[0]), action_dim=env.action_space.n, save_dir=dir, old_version="2023-12-07T17-17-43" )


states_path = 'states'
# File for states!
files = os.listdir(states_path)

episodes = 100

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



