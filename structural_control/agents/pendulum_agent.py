# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class PendulumAgent
#   @author: by Kangyao Huang
#   @created date: 24.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is an example for training a Pendulum.
"""

import numpy as np
import torch
from lib.agents.agent_ppo2 import AgentPPO2
from structural_control.envs.pendulum import PendulumEnv


class PendulumAgent(AgentPPO2):
    def __init__(self, cfg, logger, dtype, device, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training

        self.setup_env()

        super().__init__(self.cfg, self.env, self.logger, self.dtype, self.device, self.num_threads,
                         training=self.training, checkpoint=checkpoint)

    def setup_env(self):
        self.env = PendulumEnv(self.cfg, flat_observation=False)

        observation_flat_dim = 0
        for k, v in self.env.task.get_observation(self.env.physics).items():
            observation_flat_dim += v.shape[0]
        self.env.state_dim = observation_flat_dim
        self.env.action_dim = self.env.action_spec().shape[0]

    def visualize_agent(self, save_video=False):
        env = self.env

        from dm_control import viewer
        def policy_fn(time_step):
            def process_dm_ctrl_observation(time_step):
                """ Flatten the dm_control observation. """
                observation_flatten = np.array([])
                for k in time_step.observation:
                    if time_step.observation[k].shape:
                        observation_flatten = np.concatenate((observation_flatten, time_step.observation[k].flatten()))
                    else:
                        observation_flatten = np.concatenate(
                            (observation_flatten, np.array([time_step.observation[k]])))
                reward = time_step.reward
                done = time_step.last()
                return observation_flatten, reward, done

            observation, reward, done = process_dm_ctrl_observation(time_step)
            state = observation
            state_var = torch.tensor(state).unsqueeze(0)
            action = self.policy_net(state_var)[0][0].detach().numpy()
            return action

        viewer.launch(environment_loader=env, policy=policy_fn)