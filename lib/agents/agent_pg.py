# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class AgentPG
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 01.Nov.2022
#   @changes: Changed same variables' name
# ------------------------------------------------------------------------------------------------------------------- #

from lib.agents.agent import Agent
import lib.core.torch_wrapper as torper
import time
import torch


class AgentPG(Agent):
    """
        Policy Gradient Agent.
    """
    def __init__(self, tau=0.95, optimizer_policy=None, optimizer_value=None,
                 optim_num_epoches=1, value_optim_num_iter=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.optim_num_epochs = optim_num_epoches
        self.value_optim_num_iter = value_optim_num_iter # value optimizer number of iteration?

    def update_value(self, states, returns):
        """ Update Critic """
        for _ in range(self.value_optim_num_iter):
            value_predict = self.value_net(self.trans_value(states))
            value_loss = (value_predict - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def update_policy(self, states, actions, returns, advantages, exps):
        """ Update Policy """
        # use a2c by default
        ind = exps.nonzero().squeeze(1)
        for _ in range(self.optim_num_epochs):
            self.update_value(states, returns)
            log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
            policy_loss = - (log_probs * advantages[ind]).mean()
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

    def update_params(self, batch):
        t_start = time.time()
        torper.to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)

        with torper.to_eval(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states))

        """get advantage estimation from the trajectories"""
        advantages, returns = torper.estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        self.update_policy(states, actions, returns, advantages, exps)

        return time.time() - t_start
