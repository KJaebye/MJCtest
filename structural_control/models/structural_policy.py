# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class StructuralPolicy
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 11.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.core.actor import Policy
from lib.models.mlp import MLP
from structural_control.models.gnn import GNNSimple
from lib.core.running_norm import RunningNorm
from lib.utils.tools import init_fc_weights
from lib.core.distributions import DiagGaussian
import torch
import numpy as np


class StruturalPolicy(Policy):
    def __init__(self, cfg_spec, agent):
        super().__init__()
        self.cfg_spec = cfg_spec
        self.agent = agent
        self.observation_flat_dim = agent.observation_flat_dim
        # self.state_dim = agent.state_dim
        self.action_dim = agent.action_dim

        self.norm = RunningNorm(self.observation_flat_dim)
        cur_dim = self.observation_flat_dim
        if 'control_pre_mlp' in cfg_spec:
            self.pre_mlp = MLP(cur_dim, cfg_spec['control_pre_mlp'], cfg_spec['htype'])
            cur_dim = self.pre_mlp.output_dim
        else:
            self.pre_mlp = None
        if 'control_gnn_spec' in cfg_spec:
            self.gnn = GNNSimple(cur_dim, cfg_spec['control_gnn_spec'])
            cur_dim = self.gnn.output_dim
        else:
            self.gnn = None
        if 'control_mlp' in cfg_spec:
            self.mlp = MLP(cur_dim, cfg_spec['control_mlp'], cfg_spec['htype'])
            cur_dim = self.mlp.output_dim

        self.action_mean = torch.nn.Linear(cur_dim, self.action_dim)
        init_fc_weights(self.action_mean)
        self.action_log_std = \
            torch.nn.Parameter(torch.ones(1, self.action_dim) * self.cfg_spec['control_log_std'],
                               requires_grad=not cfg_spec['fix_control_std'])

    def batch_data(self, x):
        pass

    def forward(self, x):
        x = self.norm(x)
        if self.pre_mlp is not None:
            x = self.pre_mlp(x)
        if self.gnn is not None:
            x = self.gnn(x, edges)
        if self.mlp is not None:
            x = self.mlp(x)

        action_mean = self.action_mean(x)
        action_std = self.action_log_std.expand_as(action_mean).exp()

        dist = DiagGaussian(action_mean, action_std)
        return dist, x[0][0].device

    def select_action(self, x, mean_action=False):
        dist, device = self.forward(x)
        action = dist.mean_sample() if mean_action else dist.sample()
        action.to(device)
        return action

    def get_log_prob(self, x, actions):
        actions = torch.cat(actions)
        x = torch.cat(x)
        dist, device = self.forward(x)

        action_log_prob = dist.log_prob(actions)
        # print(action_log_prob)
        # print(action_log_prob.shape)
        # action_log_prob_cum = torch.cumsum(action_log_prob, dim=0)
        # action_log_prob = torch.cat([action_log_prob_cum[[0]], action_log_prob_cum[1:] - action_log_prob_cum[:-1]])
        # print(action_log_prob.shape)

        return action_log_prob
