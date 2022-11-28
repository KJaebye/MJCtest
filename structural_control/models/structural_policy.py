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
        self.action_dim = agent.action_dim

        self.norm = RunningNorm(self.observation_flat_dim)
        cur_dim = self.observation_flat_dim
        if 'pre_mlp' in cfg_spec:
            self.pre_mlp = MLP(cur_dim, cfg_spec['pre_mlp'], cfg_spec['htype'])
            cur_dim = self.pre_mlp.output_dim
        else:
            self.pre_mlp = None
        if 'gnn_spec' in cfg_spec:
            self.gnn = GNNSimple(cur_dim, cfg_spec['gnn_spec'])
            cur_dim = self.gnn.output_dim
        else:
            self.gnn = None
        if 'mlp' in cfg_spec:
            self.mlp = MLP(cur_dim, cfg_spec['mlp'], cfg_spec['htype'])
            cur_dim = self.mlp.output_dim

        self.action_mean = torch.nn.Linear(cur_dim, self.action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = torch.nn.Parameter(torch.ones(1, self.action_dim) * self.cfg_spec['log_std'])

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
        # print(action_mean)
        dist = DiagGaussian(action_mean, action_std)
        return dist, x[0][0].device

    def select_action(self, x, use_mean_action=False):
        dist, device = self.forward(x)
        action = dist.mean_sample() if use_mean_action else dist.sample()
        action.to(device)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        dist, device = self.forward(x)
        action_log_prob = dist.log_prob(actions)
        return action_log_prob
