# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class HopperAgent
#   @author: by Kangyao Huang
#   @created date: 07.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.agents.agent_ppo import AgentPPO
from structural_control.envs.hopper import HopperEnv

"""
    This agent is an example for training a robot.
"""

class HopperAgent(AgentPPO):
    def __init__(self, cfg, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.setup_env()

        super(AgentPPO).__init__()

    def setup_env(self):
        self.env = HopperEnv(self.cfg)
