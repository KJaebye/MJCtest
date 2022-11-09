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
from lib.core.memory import Memory

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

        super(AgentPPO).__init__(env=self.env, dtype=self.dtype, cfg=self.cfg, device=self.device,
                                 policy_net=self.policy_net, value_net=self.value_net, gamma=self.gamma,
                                 custom_reward=None, logger_cls=LoggerRL, traj_cls=TrajBatch,
                                 logger_kwargs=None, end_reward=False, running_state=None,
                                 num_threads=self.num_threads)

    def setup_env(self):
        self.env = HopperEnv(self.cfg)

    def sample_worker(self, pid, queue, thread_batch_size, mean_action, render):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)

        while logger.num_steps < thread_batch_size:
            time_step = self.env.reset()
            if self.running_state is not None:
                time_step = self.running_state(time_step)
                logger.start_episode(self.env)
                self.pre_episode()

                for t in range(self.cfg.max_timesteps):
                    pass







