import gym
from GameState import GameState
# We dont need to act on all of the frames so we can skip them beween actions
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):

        """Repeat action, and sum reward"""
        total_reward = 0
        
        for i in range(self._skip):

            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)

            # Add the rewards together
            total_reward += reward
            if done:
                break

        # Update previous game state
        self.env.prevState = GameState(self.pyboy)
        return obs, total_reward, done,  info