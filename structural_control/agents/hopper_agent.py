# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class HopperAgent
#   @author: by Kangyao Huang
#   @created date: 07.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is an example for training a robot.
"""
import pickle
import time

import torch
from lib.agents.agent_ppo import AgentPPO
from lib.core.logger_rl import LoggerRL
from lib.core.traj_batch import TrajBatch
from lib.core import torch_wrapper as torper
from lib.core.common import estimate_advantages
from structural_control.envs.hopper import HopperEnv
from structural_control.models.structural_policy import StruturalPolicy
from structural_control.models.structural_critic import StructuralValue
from torch.utils.tensorboard import SummaryWriter
from lib.core.memory import Memory


def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

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
        self.setup_param_scheduler()

        super(AgentPPO).__init__(env=self.env, dtype=self.dtype, cfg=self.cfg, device=self.device,
                                 policy_net=self.policy_net, value_net=self.value_net, gamma=self.gamma,
                                 logger_cls=LoggerRL, traj_cls=TrajBatch, logger_kwargs=None,
                                 running_state=None, num_threads=self.num_threads)
    def sample_worker(self, pid, queue, thread_batch_size, mean_action, render):


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

    def setup_param_scheduler(self):
        self.scheduled_params = {}
        for name, specs in self.cfg.scheduled_params.items():
            if specs['type'] == 'step':
                self.scheduled_params[name] = torper.StepParamScheduler(specs['start_val'], specs['step_size'],
                                                                       specs['gamma'], specs.get('smooth', False))
            elif specs['type'] == 'linear':
                self.scheduled_params[name] = torper.LinearParamScheduler(specs['start_val'], specs['end_val'],
                                                                         specs['start_epoch'], specs['end_epoch'])

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
            with torper.to_cpu(self.policy_net, self.value_net):
                model_checkpoint = \
                {
                    'policy_dict': self.policy_net.state_dict(),
                    'value_dict': self.value_net.state_dict(),
                    'running_state': self.running_state,
                    'loss_iter': self.loss_iter,
                    'best_rewards': self.best_rewards,
                    'epoch': epoch
                }
                pickle.dump(model_checkpoint, open(checkpoint_path, 'wb'))

        additional_saves = self.cfg.agent_spec.get('additional_saves', None)
        if (self.cfg.save_model_interval > 0 and (epoch+1) % self.cfg.save_model_interval == 0) or \
            (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.tb_logger.flush()
            save('%s/epoch_%04d.p' % (self.cfg.model_dir, epoch+1))
        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.critical(f'Save best checkpoint with rewards {self.best_rewards:.2f}')
            save('%s/best.p' % self.cfg.model_dir)


    def pre_epoch_update(self, epoch):
        for param in self.scheduled_params.values():
            param.set_epoch(epoch)

    def optimize(self, epoch):
        self.pre_epoch_update(epoch)
        info = self.optimize_policy(epoch)
        self.log_optimize_policy(epoch, info)


    def optimize_policy(self, epoch):
        """
        Generate multiple trajectories that reach the minimum batch_size.
        :param epoch:
        :return:
        """
        t_0 = time.time()
        # sample a batch of data
        batch, log = self.sample(self.cfg.min_batch_size)

        # update networks
        t_1 = time.time()
        self.update_params(batch)
        t_2 = time.time()

        # evaluate policy
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t_3 = time.time()

        info = {
            'log': log, 'log_eval': log_eval, 'sample_time': t_1-t_0, 'update_time': t_2-t_1,
            'eval_time': t_3-t_2, 'total_time': t_3-t_0
        }
        return info

    def update_params(self, batch):
        torper.to_train(*self.update_modules)
        states = tensorfy(batch.states, self.device)
        actions = tensorfy(batch.actions, self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        with torper.to_eval(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = 10000
                for i in range(1, len(states), chunk):
                    states_i = states[i:min(i+chunk, len(states))]
                    values_i = self.value_net(self.trans_value(states_i))
                    values.append(values_i)
                values = torch.cat(values)

        # get advantage from the trajectories
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        if self.cfg.agent_spec.get('reinforce', False):
            advantages = returns.clone()

        self.update_policy(states, actions, returns, advantages, exps)
        return


    def update_policy(self, states, actions, returns, advantages, exps):











