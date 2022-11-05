# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class StructuralControlAgent
#   @author: by Kangyao Huang
#   @created date: 03.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.agents.agent_ppo import AgentPPO
from structural_control.envs import domain_dict
from structural_control.envs import task_dict
from dm_control.rl.control import Environment


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
        self.loss_iter = 0

        self.setup_env()


        super(AgentPPO).__init__()


    def setup_env(self):

