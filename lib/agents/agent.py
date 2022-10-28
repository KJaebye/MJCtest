# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Agent
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 27.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import platform
import os
from lib.core.memory import Memory
from lib.core.torch import *


if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:
    def __init__(self, env, policy_net, value_net, dtype, logger, cfg, device, gamma,
                 custom_reward=None, end_reward=False, running_state=None,
                 num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.gamma = gamma
        self.custom_reward = custom_reward
        self.end_reward = end_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        """ Sample min_batch_size of data. """
        self.seed_worker(pid)
        memory = Memory()

        while self.logger.num_steps < min_batch_size:
            state = self.env.reset()
            # preprocess state if needed
            if self.running_state is not None:
                state = self.running_state(state)
            self.logger.start_episode(self.env)
            self.pre_episode()

            for _ in range(self.cfg.max_timesteps):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                # sample an action
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(trans_out, use_mean_action)[0].numpy()
                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                # apply this action and get env feedback
                next_state, env_reward, done, info = self.env.step(action)

                # preprocess state if needed
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, env_reward, info)
                    reward = c_reward
                else:
                    c_reward, c_info = .0, np.array([.0])
                    reward = env_reward
                # add env reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging



    def pre_episode(self):
        return

    def seed_worker(self, pid):
        if pid > 0:
