# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class HopperAgent
#   @author: by Kangyao Huang
#   @created date: 07.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is an example for training a robot.
"""
import math
import pickle
import time

import numpy as np
import torch
import mujoco_viewer
from lib.agents.agent_ppo import AgentPPO
from lib.core.logger_rl import LoggerRL
from lib.core.traj_batch import TrajBatch
from lib.core import torch_wrapper as torper
from lib.core.common import estimate_advantages
from lib.core.memory import Memory
from lib.utils import tools
from structural_control.envs.hopper import HopperEnv
from structural_control.models.structural_policy import StruturalPolicy
from structural_control.models.structural_critic import StructuralValue
from torch.utils.tensorboard import SummaryWriter



def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class HopperAgent(AgentPPO):
    def __init__(self, cfg, logger, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.action_dim = None
        self.observation_dim = None
        self.observation_flat_dim = 0
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.t_start = time.time()
        self.total_steps = 0

        self.setup_env()
        self.setup_tb_logger()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.setup_param_scheduler()
        self.save_best_flag = False
        if checkpoint != 0:
            self.load_checkpoint(checkpoint)

        super().__init__(env=self.env, dtype=self.dtype, cfg=self.cfg, device=self.device,
                         policy_net=self.policy_net, value_net=self.value_net,
                         gamma=cfg.gamma, tau=cfg.tau,
                         logger_cls=LoggerRL, traj_cls=TrajBatch,
                         logger_kwargs=None, running_state=None, num_threads=self.num_threads,
                         optimizer_policy=self.optimizer_policy, optimizer_value=self.optimizer_value,
                         optim_num_epoches=cfg.num_optim_epoch,
                         clip_epsilon=cfg.clip_epsilon,
                         policy_grad_clip=[(self.policy_net.parameters(), 40)],
                         use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size,
                         mini_batch_size=cfg.mini_batch_size)

    def sample_worker(self, pid, queue, thread_batch_size, mean_action, render):
        """
        Sample min_batch_size of data.
        :param pid: work index
        :param queue: for multiprocessing
        :param thread_batch_size: how many batches of data should be collected by one worker
        :param mean_action: bool type
        :param render: bool type
        :return:

        time_step is the instantiation of dm_env.TimeStep
        """
        self.seed_worker(pid)
        memory = Memory()
        logger_rl = self.logger_cls(**self.logger_kwargs)

        # sample a batch of data
        while logger_rl.num_steps < thread_batch_size:
            time_step = self.env.reset()
            cur_state = torper.tensor([tools.get_state(time_step.observation)], device=self.device)

            # preprocess state if needed
            if self.running_state is not None:
                time_step.observation = self.running_state(time_step.observation)
            logger_rl.start_episode(self.env)
            self.pre_episode()

            # sample an episode
            while not time_step.last():
                # use trans_policy before entering the policy network
                cur_state = self.trans_policy(cur_state)

                # sample an action
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(cur_state, use_mean_action).numpy()
                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)

                # apply this action and get env feedback
                time_step = self.env.step(action)
                reward = time_step.reward
                next_state = torper.tensor([tools.get_state(time_step.observation)], device=self.device)

                # add end reward
                if self.end_reward and time_step.last():
                    reward += self.env.end_reward

                # preprocess state if needed
                if self.running_state is not None:
                    next_state = self.running_state(next_state)


                # record reward
                logger_rl.step(reward)
                mask = 0 if time_step.last() else 1
                exp = 1 - use_mean_action
                self.push_memory(memory, cur_state, action, next_state, reward, mask, exp)
                cur_state = next_state

                # only render the first worker pid 0
                """ 
                    Only one glfw window can be displayed. However, there are "self.num_threads" number of
                    workers created and running simultaneously when we use multiprocessing method. Thus,
                    when user sets parameter render to be True and num_threads > 1, we only display the first
                    worker's action in simulator.
                """
                if pid == 0 and render:
                    viewer = mujoco_viewer.MujocoViewer(self.env.physics, self.env.physics.data)
                    viewer.render()
                if time_step.last():
                    break

            logger_rl.end_episode()
        logger_rl.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger_rl])
        else:
            return memory, logger_rl

    def setup_env(self):
        self.env = HopperEnv(self.cfg, flat_observation=False)
        """ observation specs and dimension """
        self.observation_dim = len(self.env.observation_spec())
        # print(self.env.observation_spec())
        # print(self.observation_dim)
        """ observation flatten dimension """
        for k, v in self.env.task.get_observation(self.env.physics).items():
            self.observation_flat_dim += v.shape[0]
        # print(self.env.task.get_observation(self.env.physics))
        # print(self.observation_flat_dim)
        """ action specs and dimension """
        self.action_dim = self.env.action_spec().shape[0]
        # print(self.env.action_spec())
        # print(self.action_dim)

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

    def setup_tb_logger(self):
        self.tb_logger = SummaryWriter(self.cfg.tb_dir) if self.training else None
        self.best_reward = -1000
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
        self.best_reward = model_checkpoint.get['best_reward', self.best_reward]
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
                        'best_reward': self.best_reward,
                        'epoch': epoch
                    }
                pickle.dump(model_checkpoint, open(checkpoint_path, 'wb'))

        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the best checkpoint with rewards {self.best_reward:.2f}')
            save('%s/best.p' % self.cfg.model_dir)
            save('%s/checkpoint_%04d.p' % (self.cfg.model_dir, epoch + 1))

    def pre_epoch_update(self, epoch):
        for param in self.scheduled_params.values():
            param.set_epoch(epoch)

    def optimize(self, epoch):
        """
        Optimize and part of logging.
        :param epoch:
        :return:
        """
        self.pre_epoch_update(epoch)
        self.logger.info('#------------------------ Iteration {} --------------------------#'.format(epoch))
        log, log_eval = self.optimize_policy(epoch)

        t_cur = time.time()
        if log_eval.episode_reward > self.best_reward:
            self.best_reward = log_eval.episode_reward
            self.save_best_flag = True
            self.logger.critical('Get the best episode reward: {}'.format(self.best_reward))
            self.save_checkpoint(epoch)

        else:
            self.save_best_flag = False
            self.logger.info('Average episode reward: {}'.format(log_eval.episode_reward))

        self.logger.info('Total time: {}'.format(t_cur - self.t_start))
        self.total_steps += self.cfg.min_batch_size
        self.logger.info('{} total steps have happened'.format(self.total_steps))

        self.tb_logger.add_scalar('train_R_avg ', log.avg_reward, epoch)
        self.tb_logger.add_scalar('train_R_eps_avg', log.avg_episode_reward, epoch)
        self.tb_logger.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, epoch)
        self.tb_logger.add_scalar('exec_R_avg', log_eval.avg_exec_reward, epoch)
        self.tb_logger.add_scalar('exec_R_eps_avg', log_eval.avg_exec_episode_reward, epoch)

    def optimize_policy(self, epoch):
        """
        Generate multiple trajectories that reach the minimum batch_size.
        :param epoch:
        :return:
        """
        t_0 = time.time()
        # sample a batch of data
        batch, log = self.sample(self.cfg.min_batch_size)
        t_1 = time.time()
        self.logger.info('Sample time: {}'.format(t_1 - t_0))

        # update networks
        self.update_params(batch)
        t_2 = time.time()
        self.logger.info('Update time: {}'.format(t_2 - t_1))

        # evaluate policy
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t_3 = time.time()
        self.logger.info('Evaluation time: {}'.format(t_3 - t_2))

        # info = {
        #     'log': log, 'log_eval': log_eval, 'sample_time': t_1 - t_0, 'update_time': t_2 - t_1,
        #     'eval_time': t_3 - t_2, 'total_time': t_3 - t_0
        # }
        return log, log_eval

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
                    states_i = states[i:min(i + chunk, len(states))]
                    values_i = self.value_net(self.trans_value(states_i))
                    values.append(values_i)
                values = torch.cat(values)

        # get advantage from the trajectories
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        if self.cfg.agent_spec.get('reinforce', False):
            advantages = returns.clone()

        self.update_policy(states, actions, returns, advantages, exps)
        return

    def get_perm_batch_design(self, states):
        inds = [[], [], []]
        for i, x in enumerate(states):
            use_transform_action = x[2]
            inds[use_transform_action.item()].append(i)
        perm = np.array(inds[0] + inds[1] + inds[2])
        return perm, torper.LongTensor(perm).to(self.device)

    def update_policy(self, states, actions, returns, advantages, exps):
        """
        Update policy.
        :param states:
        :param actions:
        :param returns:
        :param advantages:
        :param exps:
        :return:
        """
        with torper.to_eval(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    actions_i = actions[i:min(i + chunk, len(states))]
                    fixed_log_probs_i = self.policy_net.get_log_prob(self.trans_policy(states_i), actions_i)
                    fixed_log_probs.append(fixed_log_probs_i)
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)

        self.logger.info('| %11s | %11s | %11s | %11s| %11s|' % ('surr', 'kl', 'ent', 'vf_loss', 'weight_l2'))

        for _ in range(self.optim_num_epochs):
            if self.use_mini_batch:
                perm_np = np.arange(num_state)
                np.random.shuffle(perm_np)
                perm = torper.LongTensor(perm_np).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = \
                    tools.index_select_list(states, perm_np), \
                    tools.index_select_list(actions, perm_np), \
                    returns[perm].clone(), \
                    advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), \
                    exps[perm].clone()

                # design?
                if self.cfg.agent_spec.get('batch_design', False):
                    perm_design_np, perm_design = self.get_perm_batch_design(states)
                    states, actions, returns, advantages, fixed_log_probs, exps = \
                        tools.index_select_list(states, perm_design_np), \
                        tools.index_select_list(actions, perm_design_np), \
                        returns[perm_design].clone(), \
                        advantages[perm_design].clone(), \
                        fixed_log_probs[perm_design].clone(), \
                        exps[perm_design].clone()

                optim_iter_num = int(math.floor(num_state / self.mini_batch_size))
                for i in range(optim_iter_num):
                    index = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, num_state))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        states[index], actions[index], advantages[index], returns[index], fixed_log_probs[index], exps[
                            index]
                    self.update_value(states_b, returns_b)
                    self.surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b)
                    self.optimizer_policy.zero_grad()
                    self.surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                index = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                self.surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs)
                self.optimizer_policy.zero_grad()
                self.surr_loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()

                # logging
                # self.logger.info('| %10.8f | %10.8f | %10.4f | %10.4f | %10.4f |' %
                # (self.surr_loss, kl_epoch, entropy_epoch, vf_epoch, weight_epoch))

                # self.logger.info('Learning rate: {}'.format(self.optimizer_policy.state_dict()['param_groups'][0]['lr']))
                # # self.logger.info('KL value: {}'.format(self.))
                # self.logger.info('Surrogate loss: {}'.format(self.surr_loss))

    def ppo_loss(self, states, actions, advantages, fixed_log_probs):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr_1 = ratio * advantages
        surr_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = - torch.min(surr_1, surr_2).mean()
        return surr_loss
