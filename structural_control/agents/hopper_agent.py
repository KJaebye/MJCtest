# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class HopperAgent
#   @author: by Kangyao Huang
#   @created date: 07.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is an example for training a robot.
"""

from lib.agents.agent_ppo import AgentPPO
from structural_control.envs.hopper import HopperEnv
from lib.core.logger_rl import LoggerRL
from lib.core.traj_batch import TrajBatch
from structural_control.models.structural_policy import StruturalPolicy
from lib.core.memory import Memory

class HopperAgent(AgentPPO):
    def __init__(self, cfg, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.action_dim = None
        self.observation_dim = None
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.setup_env()

        super(AgentPPO).__init__(env=self.env, dtype=self.dtype, cfg=self.cfg, device=self.device,
                                 policy_net=self.policy_net, value_net=self.value_net, gamma=self.gamma,
                                 logger_cls=LoggerRL, traj_cls=TrajBatch, logger_kwargs=None,
                                 running_state=None, num_threads=self.num_threads)

    def setup_env(self):
        self.env = HopperEnv(self.cfg)
        self.observation_dim = len(self.env.observation_spec())
        self.action_dim = len(self.env.action_spec())
        self.running_state = None

    def setup_policy(self):
        self.policy_net = StruturalPolicy(self.cfg.policy_specs, self)

    def setup_value(self):
        self.value_net =

    def setup_logger(self):

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, int):


    def save_checkpoint(self, epoch):
        def save(checkpoint_path):


    def pre_epoch_update(self, epoch):


    def optimize(self, epoch):


    def optimize_policy(self, epoch):


    def update_policy(self, states, actions, returns, advantages, exps):











