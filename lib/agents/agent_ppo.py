# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class AgentPPO
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 02.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
import math
import torch
import numpy as np

from lib.agents.agent_pg import AgentPG
import lib.core.torch_wrapper as torper


class AgentPPO(AgentPG):
    def __init__(self, clip_epsilon=0.2, mini_batch_size=64, use_mini_batch=False,
                 policy_grad_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch = use_mini_batch
        self.policy_grad_clip = policy_grad_clip

    def update_policy(self, states, actions, returns, advantages, epoch):
        """ Update the policy """
        with torper.to_eval(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)

        for _ in range(self.optim_num_epochs):
            if self.use_mini_batch:
                # randomly arrange data
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = torper.LongTensor(perm).to(self.device)
                states, actions, returns, advantages, fixed_log_probs = \
                    states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone()

                optim_iter_num = int(math.floor(states.shape[0] / self.mini_batch_size))
                for i in range(optim_iter_num):
                    # index denotes data range of the current batch
                    index = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, states.shape[0]))

                    # intercept the current batch data
                    states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b = \
                        states[index], actions[index], returns[index], \
                        advantages[index], fixed_log_probs[index]

                    # update value by using the current batch
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs)
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()

    def clip_policy_grad(self):
        if self.policy_grad_clip is not None:
            for params, max_norm in self.policy_grad_clip:
                torch.nn.utils.clip_grad_norm_(params, max_norm)

    def ppo_loss(self, states, actions, advantages, fixd_log_probs):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)
        ratio = torch.exp(log_probs - fixd_log_probs)
        surr_1 = ratio * advantages
        surr_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = - torch.min(surr_1, surr_2).mean()
        return surr_loss
