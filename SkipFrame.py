import gym
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

            # Accumulate reward and repeat the same action but without altering button presses
            # If we used the normal step function, it would register button presses
            # And the agent could not move at all
            obs, reward, done, info = self.env.step(action)
            # Add the rewards together
            total_reward += reward

            if done:
                break
        return obs, total_reward, done,  info