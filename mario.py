from pyboy import WindowEvent
from gym.spaces import Discrete
import numpy as np
import difflib

class Mario:
  def __init__(self, state_dim, action_dim, save_dir ):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.save_dir = save_dir

    self.exploration_rate = 1
    self.exploration_rate_decay = 0.99999975
    self.exploration_rate_min = 0.1
    self.curr_step = 0

    # Define the different action of our agent 
    self._DO_NOTHING = WindowEvent.PASS
    self._buttons = [
      WindowEvent.PRESS_ARROW_RIGHT,
      WindowEvent.PRESS_BUTTON_A,
      WindowEvent.RELEASE_BUTTON_A
      ]

    self._buttons_release = [
      WindowEvent.RELEASE_ARROW_RIGHT,
      WindowEvent.RELEASE_BUTTON_A
    ]
    self.actions = [self._DO_NOTHING] + self._buttons
    self.action_space = Discrete(len(self.actions))

    # Q_learning value
    self.Q_learningFunction_file="Q_function.npy"
    self.learning_rate = 0.8
    self.discount_factor = 0.95
    self.defaultActionDict={key: 0 for key in self.actions}
    self.q_dict = {}
 
    self.previousStates = []
    # Define our policy by random policy if no policy is given
    
  def Q_learning(self,current_state,next_state,action,reward):
    currentStateKey= current_state.tobytes()
    nextStateKey = next_state.tobytes()
    current_q_values = self.q_dict.get(currentStateKey, self.defaultActionDict)
    current_q_values[action] = (1 - self.learning_rate) * current_q_values[action] + self.learning_rate * (reward + self.discount_factor * max(self.q_dict.get(nextStateKey, self.defaultActionDict).values()))
    print(max(self.q_dict.get(nextStateKey, self.defaultActionDict).values()))
    self.q_dict[currentStateKey] = current_q_values
    return

  def act(self, state):
    stateKey= state.tobytes()
    actionDict = self.q_dict.get(stateKey, self.defaultActionDict)
    if np.random.rand() < self.exploration_rate_min:  # Exploration
      action = np.random.choice(list(actionDict.keys()))
    else:  # Exploitation
      print(actionDict.values())
      action = max(actionDict,key=actionDict.get )  
    return action

  def saveQ_function(self):
    np.save(self.Q_learningFunction_file, self.q_dict)
    my_string = str(self.q_dict)
    return

  def loadQ_function(self):
    try:
      data=np.load(self.Q_learningFunction_file, allow_pickle="TRUE")
      self.q_dict = data.item()
      return True
    except  :
      return False


      
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
