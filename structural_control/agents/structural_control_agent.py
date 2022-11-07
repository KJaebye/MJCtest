# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class StructuralControlAgent
#   @author: by Kangyao Huang
#   @created date: 03.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.agents.agent_ppo import AgentPPO
from structural_control.envs.centipede import CentipedeEnv
from lib.core.logger_rl import LoggerRL
from lib.core.traj_batch import TrajBatch

class StructuralControlAgent(AgentPPO):
    """
        StructuralControlAgent is the core for structural control experiments,
        which inherits attributes from AgentPPO. Except for optimization, sampling,
        forward and backward operations, this class also provides some other main
        functions below:
            1. Get training settings by input cfg.
            2. Set environment configs and interactions.
            3. Save checkpoints for training.
    """

    def __init__(self, cfg, dtype, device, seed, num_threads, training=True, checkpoint=0):
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
                                 custom_reward=None, logger_cls=LoggerRL, traj_cls=TrajBatch,
                                 logger_kwargs=None, end_reward=False, running_state=None,
                                 num_threads=self.num_threads)

    def setup_env(self):
        env = CentipedeEnv(self.cfg)
