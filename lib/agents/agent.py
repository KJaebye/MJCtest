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
from lib.core.traj_batch import TrajBatch
from lib.core import torch_wrapper as torper
from lib.utils import tools

if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")

os.environ["OMP_NUM_THREADS"] = "1"


class Agent:
    def __init__(self, env, policy_net, value_net, dtype, cfg, device, gamma,
                 logger_cls=LoggerRL, logger_kwargs=None, end_reward=False,
                 running_state=None, traj_cls=TrajBatch, num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.cfg = cfg
        self.device = device
        self.gamma = gamma

        self.end_reward = end_reward
        self.running_state = running_state

        self.num_threads = num_threads
        self.noise_rate = 1.0

        self.traj_cls = traj_cls
        self.logger_cls = logger_cls
        self.logger_kwargs = dict() if logger_kwargs is None else logger_kwargs

        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]

    def sample(self, min_batch_size, mean_action=False, render=False, nthreads=None):
        """
        Sample a batch of data.
        :param min_batch_size: minimum batch size
        :param mean_action: bool type
        :param render: bool type
        :param nthreads: number of threads
        :return:
        """
        if nthreads is None:
            nthreads = self.num_threads
        t_start = time.time()
        torper.to_eval(*self.sample_modules)

        with torper.to_cpu(*self.sample_modules):
            with torch.no_grad():
                # multiprocess sampling
                thread_batch_size = int(math.floor(min_batch_size / nthreads))
                queue = multiprocessing.Queue()
                memories = [None] * nthreads
                loggers = [None] * nthreads

                for i in range(nthreads - 1):
                    worker_args = (i + 1, queue, thread_batch_size, mean_action, render)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                # save sample data from first worker pid 0
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render)
                # save sample data from other workers
                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger

                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

        logger.sample_duration = time.time() - t_start
        return traj_batch, logger

    def sample_worker(self, pid, queue, thread_batch_size, mean_action, render):
        """
        Sample min_batch_size of data.
        :param pid: work index
        :param queue: for multiprocessing
        :param thread_batch_size: how many batches of data should be collected by one worker
        :param mean_action: bool type
        :param render: bool type
        :return:

        time_step is the instantiation of dm_env.TimeStep
        """
        self.seed_worker(pid)
        memory = Memory()
        logger_rl = self.logger_cls(**self.logger_kwargs)

        # sample a batch data
        while logger_rl.num_steps < thread_batch_size:
            time_step = self.env.reset()
            cur_state = torper.tensor([tools.get_state(time_step.observation)], device=self.device)

            # preprocess state if needed
            if self.running_state is not None:
                time_step.observation = self.running_state(time_step.observation)
            logger_rl.start_episode(self.env)
            self.pre_episode()

            # sample an episode
            while not time_step.last():
                # use trans_policy before entering the policy network
                cur_state = self.trans_policy(cur_state)

                # sample an action
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(cur_state, use_mean_action)
                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)

                # apply this action and get env feedback
                time_step = self.env.step(action)
                reward = time_step.reward
                next_state = time_step.observation

                # add end reward
                if self.end_reward and time_step.last():
                    reward += self.env.end_reward

                # preprocess state if needed
                if self.running_state is not None:
                    next_state = self.running_state(next_state)

                # record reward
                logger_rl.step(reward)
                mask = 0 if time_step.last() else 1
                exp = 1 - use_mean_action
                self.push_memory(memory, cur_state, action, next_state, reward, mask, exp)
                cur_state = next_state

                # only render the first worker pid 0
                """ 
                    Only one glfw window can be displayed. However, there are "self.num_threads" number of
                    workers created and running simultaneously when we use multiprocessing method. Thus,
                    when user sets parameter render to be True and num_threads > 1, we only display the first
                    worker's action in simulator.
                """
                if pid == 0 and render:
                    ############## env.render should be replaced by mujoco native python bindings
                    self.env.render()
                if time_step.last():
                    break

            logger_rl.end_episode()
        logger_rl.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger_rl])
        else:
            return memory, logger_rl

    def seed_worker(self, pid):
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            #########
            ### env.np_random is from gym.utils
            # need use mojoco python bindings to replace gym.seeding
            if hasattr(self.env, 'np_random'):
                self.env.np_random.seed(self.env.np_random.randint(5000) * pid)

    def trans_policy(self, states):
        """transform states before going into policy net"""
        return states

    def trans_value(self, states):
        """transform states before going into value net"""
        return states

    def pre_episode(self):
        return

    def push_memory(self, memory, cur_state, action, next_state, reward, mask, exp):
        memory.push(memory, cur_state, action, next_state, reward, mask, exp)