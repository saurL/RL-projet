from pyboy import WindowEvent
from gym.spaces import Discrete
import numpy as np

class Mario:
  def __init__(self, state_dim, action_dim, save_dir):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.save_dir = save_dir

    self.exploration_rate = 1
    self.exploration_rate_decay = 0.99999975
    self.exploration_rate_min = 0.1
    self.curr_step = 0

    self._DO_NOTHING = WindowEvent.PASS
    self._buttons = [
      WindowEvent.PRESS_ARROW_RIGHT,
      WindowEvent.PRESS_BUTTON_A
      ]

    self._buttons_release = [
      WindowEvent.RELEASE_ARROW_RIGHT,
      WindowEvent.RELEASE_BUTTON_A
    ]
    self.actions = [self._DO_NOTHING] + self._buttons
    self.action_space = Discrete(len(self.actions))

  def act(self, state):

    actionIndex= np.random.randint(self.action_space.n)

                      
    return self.actions[actionIndex]
    """
    # Given a state, choose epsilon-greedy action and update the value of the step
    # EXPLORE epsilon greedy

    # Mario has 30% chance to walk right

    if np.random.rand() < self.exploration_rate:

      action_idx = np.random.randint(self.action_dim)

    # EXPLOIT
    else:
        # Implement exploitation here!!!
        '''
        state = np.array(state,dtype = float)
        action_values = 
        action_idx = torch.argmax(action_values, dim=1).item()
        '''

    # decrease exploration_rate
    self.exploration_rate *= self.exploration_rate_decay
    self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

    # increment step
    self.curr_step += 1
    return action_idx
    """
    # cache(): Each time Mario performs an action, he stores the experience to his memory.
    # His experience includes the current state, action performed, reward from the action,
    # the next state, and whether the game is done.
