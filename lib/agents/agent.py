# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Agent
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 27.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import platform
import os


if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:
    def __init__(self, env, policy_net, value_net, dtype, logger, device, gamma,
                 custom_reward=None, end_reward=False, running_state=None,
                 num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.custom_reward = custom_reward
        self.end_reward = end_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.logger = logger
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        self.seed_worker(pid)



    def seed_worker(self, pid):
        if pid > 0: