# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Agent
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 27.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import platform
import os
import time
import torch
import math
import multiprocessing
import numpy as np

from lib.core.memory import Memory
from lib.core.logger_rl import LoggerRL
from lib.core.utils import *

if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")


class Agent:
    def __init__(self, env, policy_net, device, logger_cls=LoggerRL, use_custom_reward=False,
                 running_state=None, num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.device = device

        self.running_state = running_state
        self.use_custom_reward = use_custom_reward
        self.num_threads = num_threads
        self.logger_cls = logger_cls

    def sample(self, batch_size, use_mean_action=False):
        """ Sample a batch of data.
        """
        with torch.no_grad():
            # multiprocess sampling
            thread_batch_size = int(math.floor(batch_size / self.num_threads))
            queue = multiprocessing.Queue()
            memories = [None] * self.num_threads
            loggers = [None] * self.num_threads
            slaves = []

            for i in range(self.num_threads - 1):
                slave_args = (i + 1, queue, thread_batch_size, use_mean_action)
                slaves.append(multiprocessing.Process(target=self.sample_slave, args=slave_args))
            for slave in slaves:
                slave.start()

            memory, logger = self.sample_slave(0, None, thread_batch_size, use_mean_action)
            memories[0], loggers[0] = memory, logger
            # save sample data from other slaves
            for _ in range(self.num_threads - 1):
                pid, slave_memory, slave_logger = queue.get()
                memories[pid] = slave_memory
                loggers[pid] = slave_logger

            # gather memories from all slaves
            for slave_memory in memories:
                # print(type(slave_memory))
                memory.append(slave_memory)

            batch = memory.sample()
            logger = self.logger_cls.merge(loggers, use_custom_reward=False)

            to_device(self.device, self.policy_net)
        return batch, logger

    def sample_slave(self, pid, queue, thread_batch_size, use_mean_action):
        """
        Data sampling slave.
        time_step is the instantiation of dm_env.TimeStep
        """
        # set seed
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            if hasattr(self.env, 'np_random'):
                self.env.np_random.seed(self.env.np_random.randint(5000) * pid)

        memory = Memory()
        logger_rl = self.logger_cls()

        # sample a batch of data
        while logger_rl.num_steps < thread_batch_size:
            # pre-episode process
            time_step = self.env.reset()
            logger_rl.start_episode()
            # process dm_control observation
            observation, reward, done = self.process_dm_ctrl_observation(time_step)
            state = observation
            state_var = torch.tensor(state).unsqueeze(0)

            # sample an episode
            while not time_step.last():
                # sample an action
                with torch.no_grad():
                    if use_mean_action:
                        action = self.policy_net(state_var)[0][0].numpy()
                    else:
                        action = self.policy_net.select_action(state_var)[0].numpy()

                # apply this action and get env feedback
                time_step = self.env.step(action)
                observation, reward, done = self.process_dm_ctrl_observation(time_step)
                next_state = observation
                mask = 0 if done else 1

                # record reward
                logger_rl.step(reward)

                # # if using custom reward
                # if self.use_custom_reward and self.env.custom_reward is not None:
                #     reward = self.env.custom_reward(cur_state, action)
                #     logger_rl.step_custom(reward)

                memory.push(state, action, mask, next_state, reward)
                state = next_state

                if time_step.last():
                    break

            logger_rl.end_episode()
        logger_rl.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger_rl])
        else:
            return memory, logger_rl

    def process_dm_ctrl_observation(self, time_step):
        """ Flatten the dm_control observation. """
        observation_flatten = np.array([])
        for k in time_step.observation:
            if time_step.observation[k].shape:
                observation_flatten = np.concatenate((observation_flatten, time_step.observation[k].flatten()))
            else:
                observation_flatten = np.concatenate((observation_flatten, np.array([time_step.observation[k]])))
        reward = time_step.reward
        done = time_step.last()
        return observation_flatten, reward, done
