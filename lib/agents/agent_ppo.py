
import torch
import time
import math
import numpy as np
from lib.agents.agent import Agent
from lib.core.logger_rl import LoggerRL
from lib.core.common import estimate_advantages
from lib.core.utils import *

from structural_control.networks.policy import Policy
from structural_control.networks.value import Value


class AgentPPO(Agent):
    def __init__(self, env, cfg, logger, dtype, device, num_threads, training=True):
        self.env = env
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training

        # parameters
        self.optim_num_epoch = self.cfg.optim_num_epoch
        self.batch_size = self.cfg.batch_size
        self.mini_batch_size = self.cfg.mini_batch_size
        self.eval_batch_size = self.cfg.eval_batch_size

        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        self.clip_epsilon = self.cfg.clip_epsilon
        self.l2_reg = self.cfg.l2_reg

        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        super().__init__(self.env, self.policy_net, self.device, logger_cls=LoggerRL, use_custom_reward=False,
                        running_state=None, num_threads=num_threads)

    def setup_policy(self):
        self.policy_net = Policy(self.env.state_dim, self.env.action_dim, log_std=self.cfg.policy_spec['log_std'])
        self.policy_net.to(self.device)
    def setup_value(self):
        self.value_net = Value(self.env.state_dim)
        self.value_net.to(self.device)
    def setup_optimizer(self):
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)

    def train(self, iter):
        t_0 = time.time()
        # sample a batch of data
        batch, log = self.sample(self.batch_size)
        t_1 = time.time()
        self.logger.info('Sample time: {}'.format(t_1 - t_0))

        # update networks
        self.update_params(batch, iter)
        t_2 = time.time()
        self.logger.info('Update time: {}'.format(t_2 - t_1))

        # evaluation
        _, log_eval = self.sample(self.eval_batch_size, use_mean_action=True)
        t_3 = time.time()
        self.logger.info('Evaluation time: {}'.format(t_3 - t_2))

        return log, log_eval

    def update_params(self, batch, iter):
        states = torch.from_numpy(np.stack(batch.next_state)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)
        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.mini_batch_size))
        for _ in range(self.optim_num_epoch):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                self.ppo_step(1, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b)

    def ppo_step(self, optim_value_iter, states, actions, returns, advantages, fixed_log_probs):
        """update critic"""
        for _ in range(optim_value_iter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            # print(value_loss)

            # weight decay
            for param in self.value_net.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        """update policy"""
        log_probs = self.policy_net.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        self.optimizer_policy.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 40)
        self.optimizer_policy.step()
