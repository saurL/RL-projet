import os
import numpy as np
import time, datetime
import matplotlib.pyplot as plt

import shutil
class MetricLogger:
    def __init__(self, save_dir, old_version=None):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.dataDir = save_dir / "data"
        if not os.path.exists(self.dataDir):
            os.makedirs(self.dataDir)
        self.Q_learningFunction_file = "Q_Function.npy"
        self.epsilon_file = "Epsilon.npy"
        self.episode_file="epsiode.npy"
        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_max_dist = []


        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        self.moving_avg_ep_max_dist = []

        # Current episode metric
        self.init_episode()
        self.lastEpisode=0

        # Timing
        self.record_time = time.time()
        
    # Changed to only reward
    def log_step(self, reward):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

    def saveData(self,episode,Q_function,epsilon):

        dataArray = ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards","ep_lengths","moving_avg_ep_rewards","moving_avg_ep_avg_losses","moving_avg_ep_avg_qs","moving_avg_ep_max_dist","moving_avg_ep_lengths"]
        for data in dataArray:
            np.save(self.dataDir/f"{data}.npy", getattr(self, data))
        
        self.saveEpsilon(epsilon)
        self.saveQ_function(Q_function)
        self.saveEpisode(episode)

    def log_episode(self, max_dist):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        # Added max distance for logging
        self.ep_max_dist.append(max_dist)

        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.curr_level_progress = 0
        self.curr_max_dist = 0

    def record(self, episode, epsilon, step , q_function):
        self.saveData(episode,q_function,epsilon)
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        mean_ep_max_dist = np.round(np.mean(self.ep_max_dist[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)
        self.moving_avg_ep_max_dist.append(mean_ep_max_dist)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {self.lastEpisode+episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Cumulative reward last 100 episodes {mean_ep_reward} - "
            f"Maximum distance reached {mean_ep_max_dist} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{self.lastEpisode+episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

    def saveQ_function(self,Qfunction):
        filePath = self.dataDir / self.Q_learningFunction_file
        np.save(filePath, Qfunction)

    def saveEpsilon(self,epsilon):
        filePath = self.dataDir / self.epsilon_file
        np.save(filePath, epsilon)

    def saveEpisode(self,episode):
        filePath = self.dataDir /  self.episode_file
        np.save(filePath, self.lastEpisode+episode)
    
    def getOldEpisode(self,old_version_path):
        filePath = old_version_path / "data" / self.episode_file
        return np.load(filePath)

    def getOldQ_fucntion(self,old_version_path):
        filePath =old_version_path /"data" / self.Q_learningFunction_file
        data=np.load(filePath, allow_pickle="TRUE")
        return data.item()

    def getOldEpsilon(self,old_version_path):
        filePath = old_version_path / "data" / self.epsilon_file
        return np.load(filePath)


    def loadOlderVersion(self,old_version_path):
        dataArray = ["epsiode","ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards","ep_lengths","moving_avg_ep_rewards","moving_avg_ep_avg_losses","moving_avg_ep_avg_qs","moving_avg_ep_max_dist","moving_avg_ep_lengths"]
        for data in dataArray:
            setattr(self,data,np.load(old_version_path/"data"/f"{data}.npy").tolist())
        shutil.copy(old_version_path / "log", self.save_log)
        self.lastEpisode=self.getOldEpisode(old_version_path)
        print(self.lastEpisode)
        return