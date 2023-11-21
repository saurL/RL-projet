from pyboy import PyBoy
from pyboy.openai_gym import PyBoyGymEnv
import numpy as np
from pyboy import WindowEvent
from gym.spaces import Discrete, MultiDiscrete, Box

class CustomEnv(PyBoyGymEnv):
  def __init__(self, *args, game_area_section=(0, 0, 32, 32), game_area_wrap_around=False, **kwargs):
    super().__init__(*args, **kwargs)

  # Step function for the agent including button release
  # If the agent presses a button, it is toggled
  # If the agent presses a button that is toggled, it is released  
  def step(self, action):
      info = {}

      if action == self._DO_NOTHING:
        pyboy_done = self.pyboy.tick()
      else:
        self.pyboy.send_input(action)
        pyboy_done = self.pyboy.tick()
        
        # Releases button always after it is pressed
        if self.action_type == "press":
            self.pyboy.send_input(self._release_button[action])

      # Reward using the game wrappers fitness metric
      new_fitness = self.game_wrapper.fitness
      reward = new_fitness - self.last_fitness
      self.last_fitness = new_fitness

      observation = self._get_observation()
      done = pyboy_done or self.game_wrapper.game_over()

      self.prevState = GameState(self.pyboy)

      return observation, reward, done, info
