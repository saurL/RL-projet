from pyboy import PyBoy
from pyboy.openai_gym import PyBoyGymEnv
import numpy as np
from pyboy import WindowEvent
from gym.spaces import Discrete, MultiDiscrete, Box

class CustomEnv(PyBoyGymEnv):
  def __init__(self, *args, game_area_section=(0, 0, 32, 32), game_area_wrap_around=False, **kwargs):
    super().__init__(*args, **kwargs)

  # WindowEvent.PRESS_ARROW_LEFT,

    self._DO_NOTHING = WindowEvent.PASS
    self._buttons = [
      WindowEvent.PRESS_ARROW_RIGHT,
      WindowEvent.PRESS_BUTTON_A
      ]

    self._buttons_release = [
      WindowEvent.PRESS_ARROW_RIGHT,
      WindowEvent.PRESS_BUTTON_A
    ]
    self.actions = [self._DO_NOTHING] + self._buttons
    self.action_space = Discrete(len(self.actions))