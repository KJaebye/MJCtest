# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class PendulumAgent
#   @author: by Kangyao Huang
#   @created date: 24.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is an example for training a Pendulum.
"""

import math
import pickle
import time

import numpy as np
import torch

from lib.agents.agent_ppo import AgentPPO
from lib.core.logger_rl import LoggerRL
from lib.core.common import estimate_advantages
from lib.core.memory import Memory
from lib.core.utils import *
from structural_control.envs.pendulum import PendulumEnv
from structural_control.networks.policy import Policy
from structural_control.networks.value import Value
from torch.utils.tensorboard import SummaryWriter


class PendulumAgent(AgentPPO):
    def __init__(self, cfg, logger, dtype, device, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.total_steps = 0

        self.setup_env()
        self.setup_tb_logger()
        self.save_best_flag = False

        super().__init__(self.env, cfg, logger, dtype, device, num_threads, training=True)
        if checkpoint != 0 or not training:
            self.load_checkpoint(checkpoint)

    def setup_env(self):
        # self.env = PendulumEnv(self.cfg, flat_observation=False)

        from dm_control import suite
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'random': seed})

        observation_flat_dim = 0
        for k, v in self.env.task.get_observation(self.env.physics).items():
            observation_flat_dim += v.shape[0]
        self.env.state_dim = observation_flat_dim
        self.env.action_dim = self.env.action_spec().shape[0]

    def setup_tb_logger(self):
        self.tb_logger = SummaryWriter(self.cfg.tb_dir) if self.training else None
        self.best_reward = - 1000
        self.save_best_flag = False

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, int):
            checkpoint_path = './results/%s/%s/epoch_%04d.p' % (self.cfg.domain, self.cfg.task, checkpoint)
            epoch = checkpoint
        else:
            assert isinstance(checkpoint, str)
            checkpoint_path = './results/%s/%s/%s.p' % (self.cfg.domain, self.cfg.task, checkpoint)

        model_checkpoint = pickle.load(open(checkpoint_path, "rb"))
        self.logger.critical('Loading model from checkpoint: %s' % checkpoint_path)

        self.policy_net.load_state_dict(model_checkpoint['policy_dict'])
        self.value_net.load_state_dict(model_checkpoint['value_dict'])
        self.running_state = model_checkpoint['running_state']

        if 'epoch' in model_checkpoint:
            epoch = model_checkpoint['epoch']

    def save_checkpoint(self, epoch, log, log_eval):
        def save(checkpoint_path):
            to_device(torch.device('cpu'), self.policy_net, self.value_net)
            model_checkpoint = \
                {
                    'policy_dict': self.policy_net.state_dict(),
                    'value_dict': self.value_net.state_dict(),
                    'running_state': self.running_state,
                    'best_reward': self.best_reward,
                    'epoch': epoch
                }
            pickle.dump(model_checkpoint, open(checkpoint_path, 'wb'))
            to_device(self.device, self.policy_net, self.value_net)

        cfg = self.cfg

        if cfg.save_model_interval > 0 and (epoch + 1) % cfg.save_model_interval == 0:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the interval checkpoint with rewards {self.best_reward:.2f}')
            save('%s/epoch_%04d.p' % (cfg.model_dir, epoch + 1))

        if log_eval.avg_episode_reward > self.best_reward:
            self.best_reward = log_eval.avg_episode_reward
            self.save_best_flag = True
            self.logger.critical('Get the best episode reward: {}'.format(self.best_reward))

        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the best checkpoint with rewards {self.best_reward:.2f}')
            save('%s/best.p' % self.cfg.model_dir)
            self.save_best_flag = False

    def optimize(self, iter):
        """
        Optimize and main part of logging.
        """
        t_start = time.time()
        self.logger.info('#------------------------ Iteration {} --------------------------#'.format(iter))
        log, log_eval = self.train(iter)

        t_cur = time.time()
        self.logger.info('Average TRAINING episode reward: {}'.format(log.avg_episode_reward))
        self.logger.info('Average EVALUATION episode reward: {}'.format(log_eval.avg_episode_reward))
        self.save_checkpoint(iter, log, log_eval)

        self.logger.info('Total time: {}'.format(t_cur - t_start))
        self.total_steps += self.cfg.batch_size
        self.logger.info('{} total steps have happened'.format(self.total_steps))

        self.tb_logger.add_scalar('train_R_eps_avg', log.avg_episode_reward, iter)
        self.tb_logger.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, iter)

    def update_params(self, batch, iter):
        states = torch.from_numpy(np.stack(batch.next_state)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)
        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.mini_batch_size))
        for _ in range(self.optim_num_epoch):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                self.ppo_step(1, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b, iter)

    def ppo_step(self, optim_value_iter, states, actions, returns, advantages, fixed_log_probs, iter):
        """update critic"""
        for _ in range(optim_value_iter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            self.tb_logger.add_scalar('value_loss', value_loss, iter)
            # print(value_loss)
            # # weight decay
            # for param in self.value_net.parameters():
            #     value_loss += param.pow(2).sum() * self.l2_reg
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        """update policy"""
        log_probs = self.policy_net.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        self.tb_logger.add_scalar('policy_loss', policy_surr, iter)
        self.optimizer_policy.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 40)
        self.optimizer_policy.step()

    def visualize_agent(self, num_episode=1, mean_action=False, save_video=False):
        from dm_control import viewer
        def policy_fn(time_step):
            observation, reward, done = self.process_dm_ctrl_observation(time_step)
            state_var = torch.tensor(observation).unsqueeze(0)
            action = self.policy_net(state_var)[0][0].detach().numpy()
            return action

        viewer.launch(self.env, policy=policy_fn)
