# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class AgentPPO
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 02.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.agents.agent_pg import AgentPG


class AgentPPO(AgentPG):
    def __init__(self, clip_epsilon=0.2, mini_batch_size=64, use_mini_batch=False,
                 policy_grad_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch = use_mini_batch
        self.policy_grad_clip = policy_grad_clip
