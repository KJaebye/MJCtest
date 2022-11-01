# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class AgentPG
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 01.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.agents.agent import Agent


class AgentPG(Agent):
    def __init__(self, tau=0.95, optimizer_policy=None, optimizer_value=None,
                 opt_num_epoches=1, value_opt_niter=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.opt_num_epoches = opt_num_epoches
        self.value_opt_niter = value_opt_niter # value optimizer number of iteration?

    def update_value(self, states, returns):
        """ Update Critic"""
        for _ in range(self.value_opt_niter):
            value_predict = self.value_net(self.trans_value(states))
            value_loss = (value_predict - returns).pow(2).mean()
