from pyboy import PyBoy
from pyboy.openai_gym import PyBoyGymEnv
import numpy as np
from pyboy import WindowEvent
from gym.spaces import Discrete, MultiDiscrete, Box
from GameState import GameState

class CustomEnv(PyBoyGymEnv):
  def __init__(self, *args, game_area_section=(0, 0, 32, 32), game_area_wrap_around=False, **kwargs):
    super().__init__(*args, **kwargs)

    # Flag to see if mario has respawned, used when mario dies to avoid too many minus rewards
    self.respawned = True

    # Information about last game state, used for reward function
    self.prevState = GameState(self.pyboy)


  # Step function for the agent including button release 
  def step(self, action):
    
    info = {}

    if action != self._DO_NOTHING:
        self.pyboy.send_input(action)
       
    pyboy_done = self.pyboy.tick()
    
    # Releases button always after it is pressed
    if action in [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_LEFT] :
        self.pyboy.send_input(self._release_button[action])


    # current state
    currState = GameState(self.pyboy)

    # Time reseting happens weirdly between ticks, so if the difference is larger than 1, reset clock     
    if currState.time_left - self.prevState.time_left > 1:
        self.prevState.time_left = 400

    # IF LIVES HAVE CHANGED MARIO SHOULD HAVE RESPAWNED
    if (currState.lives_left < self.prevState.lives_left):
        # Reset the distance
        self.prevState.real_x_pos = currState.real_x_pos
        # Reset the clock
        self.respawned = True

    currScore = (currState.time_left)*2 + currState.real_x_pos
    prevScore = (self.prevState.time_left)*2 + self.prevState.real_x_pos
 
    reward = currScore - prevScore

    # CHECK IF MARIO HAS DIED
    if (0<self.pyboy.get_memory_value(0xC0AC)<5):
        if self.respawned:
            # Since this is behind a flag and it occurs only once for every frame skip
            reward -= 100
            self.respawned = False

    # CHECK IF MARIO FELL OUT OF MAP
    if (self.pyboy.get_memory_value(0xC201)>183):
        if self.respawned:
            # Since this is behind a flag and it occurs only once for every frame skip
            reward -= 100
            self.respawned = False

    self.prevState = currState
    observation = self._get_observation()
    done = pyboy_done or self.game_wrapper.game_over()

    return observation, reward, done, info
  def reset(self):
    
      """ Reset (or start) the gym environment throught the game_wrapper """
      if not self._started:
          self.game_wrapper.start_game(**self._kwargs)
          self._started = True
      else:
          self.game_wrapper.reset_game()

      # Reset max level progress
      self.game_wrapper._level_progress_max = 0
      # Update previous state
      self.prevState = GameState(self.pyboy)
      # Set time left to 400
      self.prevState.time_left = 400
      # Set buttons pressed to false
      self.button_is_pressed = {button: False for button in self._buttons}
      # Mario respawned = True
      self.respawned = True
      return self._get_observation()