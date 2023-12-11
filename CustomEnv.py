from pyboy import PyBoy
from pyboy.openai_gym import PyBoyGymEnv
import numpy as np
from pyboy import WindowEvent
from gym.spaces import Discrete, MultiDiscrete, Box
from GameState import GameState


class CustomEnv(PyBoyGymEnv):
    def __init__(self, *args, game_area_section=(0, 0, 32, 32), game_area_wrap_around=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.falled = False
        # Flag to see if mario has respawned, used when mario dies to avoid too many minus rewards
        self.respawned = True

        # Information about last game state, used for reward function
        self.prevState = GameState(self.pyboy)
        # information to be able to calculate if mario got it
        self.currentStateBig = False
        self.allStateBig = [self.currentStateBig]
        self.animationOfColision = False
        self.animationOfExpansion = False
        self.marioGotHit = False
    # Step function for the agent including button release

    def step(self, action):

        info = {}

        if action != self._DO_NOTHING:
            self.pyboy.send_input(action)

        pyboy_done = self.pyboy.tick()

        # Releases button always after it is pressed
        if action in [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_LEFT,]:
            self.pyboy.send_input(self._release_button[action])

        # current state
        currState = GameState(self.pyboy)

        # Time and x pos reseting happens between ticks, so if the difference is larger than 1, reset clock and x pos
        # In order to avoid big minus reward for spawning.
        # If unnaturally large change happens, the values should be equal
        if currState.time_left - self.prevState.time_left > 1 or currState.time_left - self.prevState.time_left < -1:
            self.prevState.time_left = currState.time_left

        if currState.real_x_pos - self.prevState.real_x_pos > 10 or currState.real_x_pos - self.prevState.real_x_pos < -10:
            self.prevState.real_x_pos = currState.real_x_pos

        # IF LIVES HAVE CHANGED MARIO SHOULD HAVE RESPAWNED
        if (currState.lives_left < self.prevState.lives_left):
            # Reset the distance
            self.prevState.real_x_pos = currState.real_x_pos
            # Reset the clock
            self.respawned = True

        currScore = (currState.time_left) + currState.real_x_pos
        prevScore = (self.prevState.time_left) + self.prevState.real_x_pos

        reward = currScore - prevScore

        # CHECK IF MARIO HAS DIED
        if (0 < self.pyboy.get_memory_value(0xC0AC) < 5):
            if self.respawned:
                # Since this is behind a flag and it occurs only once for every frame skip
                reward -= 100
                self.respawned = False

        # CHECK IF MARIO FELL OUT OF MAP
        if (self.pyboy.get_memory_value(0xC201) > 183):
            if self.respawned:
                self.falled = True
                # Since this is behind a flag and it occurs only once for every frame skip
                reward -= 100
                self.respawned = False
        # CHECK IF TERE IS A RESPAWN TIMER
        if self.pyboy.get_memory_value(0xFFA6) > 0:
            if self.falled:
                self.falled = not self.falled
            else:
                self.respawned = False
                reward = 0
        # IF MARIO GOT HIT
        if self.gotHit(self._get_observation()):
            reward -= 100
            self.marioGotHit = False

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

    def gotHit(self, state):
        # we want to search in the map of the game is we can find the pixels corresponding to big or small mario
        for line in state:
            for i in range(len(line) - 1):
                current_element = line[i]
                next_element = line[i + 1]
                # patern for big mario
                if (current_element+1 == next_element and 32 <= current_element <= 40):
                    if not self.currentStateBig:
                        self.currentStateBig = True
                # patern for small mario
                if (current_element+1 == next_element and 0 <= current_element <= 8):
                    if self.currentStateBig:
                        self.currentStateBig = False

        laststatebig = self.allStateBig[-1]
        # if the state changed
        if laststatebig != self.currentStateBig:
            # at the beggining of the animation we check if we are growing or if we got hit
            # we also dont want to trigger this check while we are in the other animation
            if self.currentStateBig and not self.animationOfColision:
                self.animationOfExpansion = True

            if not self.currentStateBig and not self.animationOfExpansion:
                if not self.animationOfColision:
                    self.marioGotHit = True
                self.animationOfColision = True

        # we check if the state stop changing for 5 frame ( had fiew bug when checking for only one)
        if all(self.currentStateBig == element for element in self.allStateBig[-5:]) and (self.animationOfColision or self.animationOfExpansion):
            self.animationOfColision = False
            self.animationOfExpansion = False

        self.allStateBig.append(self.currentStateBig)

        return self.marioGotHit
