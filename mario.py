import datetime
from pathlib import Path
from pyboy import WindowEvent
from gym.spaces import Discrete
import numpy as np
import difflib
from MetricLogger import MetricLogger


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, old_version=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Metric logger
        # The progress is saved in a checkpoint folders

        self.save_dir = Path(
            save_dir) / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.save_dir.mkdir(parents=True)

        self.logger = MetricLogger(self.save_dir)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.05
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
        self.Q_learningFunction_file = "Q_function.npy"
        self.learning_rate = 0.2
        self.discount_factor = 0.95
        self.defaultActionDict = {key: 0 for key in self.actions}
        self.q_dict = {}

        self.previousStates = []
        # Define our policy by random policy if no policy is given
        # if we want to load our data Q_function from other learning state
        if (old_version):
            old_version_path = Path(save_dir) / old_version
            self.logger.loadOlderVersion(old_version_path)
            self.exploration_rate = self.logger.getOldEpsilon(old_version_path)
            self.q_dict = self.logger.getOldQ_fucntion(old_version_path)
        print(self.defaultActionDict)

    def Q_learning(self, current_state, next_state, action, reward):
        currentStateKey = current_state.tobytes()
        nextStateKey = next_state.tobytes()
        if reward == 0:
            return
        current_q_values = self.q_dict.get(
            currentStateKey, self.defaultActionDict.copy())
        current_q_values[action] = (1 - self.learning_rate) * current_q_values[action] + self.learning_rate * (
            reward + self.discount_factor * max(self.q_dict.get(nextStateKey, self.defaultActionDict).values()))
        self.q_dict[currentStateKey] = current_q_values
        return

    def showInformation(self, current_state, next_state, reward, action):
        currentStateKey = current_state.tobytes()
        nextStateKey = next_state.tobytes()
        current_q_values = self.q_dict.get(
            currentStateKey, self.defaultActionDict.copy())
        if reward < -50:
            print(reward)
            print(current_q_values)

    def act(self, state):
        stateKey = state.tobytes()
        actionDict = self.q_dict.get(stateKey, self.defaultActionDict)
        if np.random.rand() < self.exploration_rate:  # Exploration
            # decrease exploration_rate
            if self.exploration_rate > self.exploration_rate_min:
                self.exploration_rate *= self.exploration_rate_decay
                self.exploration_rate = max(
                    self.exploration_rate, self.exploration_rate_min)

            action = np.random.choice(list(actionDict.keys()))
        else:  # Exploitation

            action = max(actionDict, key=actionDict.get)

        return action

    def record(self, episode):
        self.logger.record(episode, self.exploration_rate,
                           self.curr_step, self.q_dict)
