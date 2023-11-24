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


    # New reward function as fitness had problematic death reward
    currState = GameState(self.pyboy)

    # Compare current score to previous score to get reward
    currScore = (currState.score + currState.time_left * 10) + currState.level_progress * 15
    prevScore = (self.prevState.score + self.prevState.time_left * 10) + self.prevState.level_progress * 15

    reward = currScore - prevScore

    # Now depending on data from the RAM, we should punish mario for dying to enemy or falling off the map

    # CHECK IF MARIO HAS DIED TO ENEMY
    if (0<self.pyboy.get_memory_value(0xC0AC)<5):
        #print("MARIO DIED")
        reward -= 2500
        respawned = False

    # IF LIVES HAVE CHANGED MARIO SHOULD HAVE RESPAWNED
    if (currState.lives_left != self.prevState.lives_left):
        #print("RESPAWNED")
        self.respawned = True
    
    # CHECK IF MARIO FELL OUT OF MAP
    if (self.pyboy.get_memory_value(0xC201)==185):
        if self.respawned:
            reward -= 10000
            #print("MARIO FELL DOWN")
            self.respawned = False

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

      self.pyboy.tick()
      self.prevState = GameState(self.pyboy)
      self.button_is_pressed = {button: False for button in self._buttons}
      return self._get_observation()