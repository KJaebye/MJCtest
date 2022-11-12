# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class HopperAgent
#   @author: by Kangyao Huang
#   @created date: 07.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is an example for training a robot.
"""
import pickle

import torch
from lib.agents.agent_ppo import AgentPPO
from lib.core.logger_rl import LoggerRL
from lib.core.traj_batch import TrajBatch
from lib.core import torch_wrapper as torper
from structural_control.envs.hopper import HopperEnv
from structural_control.models.structural_policy import StruturalPolicy
from structural_control.models.structural_critic import StructuralValue
from torch.utils.tensorboard import SummaryWriter
from lib.core.memory import Memory


class HopperAgent(AgentPPO):
    def __init__(self, args, cfg, logger, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.action_dim = None
        self.observation_dim = None
        self.args = args
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint

        self.setup_env()
        self.setup_logger()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()

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
        self.policy_net = StruturalPolicy(self.cfg.policy_spec, self)
        torper.to_device(self.device, self.policy_net)

    def setup_value(self):
        self.value_net = StructuralValue(self.cfg.policy_spec, self)
        torper.to_device(self.device, self.value_net)

    def setup_optimizer(self):
        # actor optimizer
        if self.cfg.policy_optimizer == 'Adam':
            self.optimizer_policy = \
                torch.optim.Adam(self.policy_net.parameters(),
                                 lr=self.cfg.policy_lr,
                                 weight_decay=self.cfg.policy_weight_decay)
        else:
            self.optimizer_policy = \
                torch.optim.SGD(self.policy_net.parameters(),
                                lr=self.cfg.policy_lr,
                                momentum=self.cfg.policy_momentum,
                                weight_decay=self.cfg.policy_weight_decay)
        # critic optimizer
        if self.cfg.value_optimizer == 'Adam':
            self.optimizer_value = \
                torch.optim.Adam(self.value_net.parameters(),
                                 lr=self.cfg.value_lr,
                                 weight_decay=self.cfg.value_weight_decay)
        else:
            self.optimizer_value = \
                torch.optim.SGD(self.value_net.parameters(),
                                lr=self.cfg.value_lr,
                                momentum=self.cfg.value_momentum,
                                weight_decay=self.cfg.value_weight_decay)

    def setup_logger(self):
        self.tb_logger = SummaryWriter(self.cfg.tb_dir) if self.args.type == 'training' else None
        self.best_rewards = -1000
        self.save_best_flag = False

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, int):
            checkpoint_path = '%s/epoch_%04d.p' % (self.cfg.model_dir, checkpoint)
            epoch = checkpoint
        else:
            assert isinstance(checkpoint, str)
            checkpoint_path = '%s/%s' % (self.cfg.model_dir, checkpoint)
        self.logger.critical('Loading model from checkpoint: %s' % checkpoint_path)
        model_checkpoint = pickle.load(open(checkpoint_path, "rb"))
        self.policy_net.load_state_dict(model_checkpoint['policy_dict'])
        self.value_net.load_state_dict(model_checkpoint['value_dict'])
        self.running_state = model_checkpoint['running_state']
        self.loss_iter = model_checkpoint['loss_iter']
        self.best_rewards = model_checkpoint.get['best_rewards', self.best_rewards]
        if 'epoch' in model_checkpoint:
            epoch = model_checkpoint['epoch']
        self.pre_epoch_update(epoch)


    def save_checkpoint(self, epoch):
        def save(checkpoint_path):


    def pre_epoch_update(self, epoch):


    def optimize(self, epoch):


    def optimize_policy(self, epoch):


    def update_policy(self, states, actions, returns, advantages, exps):











