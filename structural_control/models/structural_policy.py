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
        self.type = 'gaussian'
        self.agent = agent
        self.observation_flat_dim = agent.observation_flat_dim
        # self.state_dim = agent.state_dim
        self.action_dim = agent.action_dim

        self.control_norm = RunningNorm(self.observation_flat_dim)
        cur_dim = self.observation_flat_dim
        if 'control_pre_mlp' in cfg_spec:
            self.control_pre_mlp = MLP(cur_dim, cfg_spec['control_pre_mlp'], cfg_spec['htype'])
            cur_dim = self.control_pre_mlp.output_dim
        else:
            self.control_pre_mlp = None
        if 'control_gnn_spec' in cfg_spec:
            self.control_gnn = GNNSimple(cur_dim, cfg_spec['control_gnn_spec'])
            cur_dim = self.control_gnn.output_dim
        else:
            self.control_gnn = None
        if 'control_mlp' in cfg_spec:
            self.control_mlp = MLP(cur_dim, cfg_spec['control_mlp'], cfg_spec['htype'])
            cur_dim = self.control_mlp.output_dim

        self.control_action_mean = torch.nn.Linear(cur_dim, self.action_dim)
        init_fc_weights(self.control_action_mean)
        self.control_action_log_std = \
            torch.nn.Parameter(torch.ones(1, self.action_dim) * self.cfg_spec['control_log_std'],
                               requires_grad=not cfg_spec['fix_control_std'])

    def batch_data(self, x):
        pass

    def forward(self, x):
        x = self.control_norm(x)
        if self.control_pre_mlp is not None:
            x = self.control_pre_mlp(x)
        if self.control_gnn is not None:
            x = self.control_gnn(x, edges)
        if self.control_mlp is not None:
            x = self.control_mlp(x)
        control_action_mean = self.control_action_mean(x)
        control_action_std = self.control_action_log_std.expand_as(control_action_mean).exp()
        control_dist = DiagGaussian(control_action_mean, control_action_std)
        return control_dist, x[0][0].device

    def select_action(self, x, mean_action=False):
        control_dist, device = self.forward(x)
        control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
        control_action.to(device)
        return control_action

    def get_log_prob(self, x, action):
        x = torch.cat(x)
        action = torch.cat(action)
        control_dist, device = self.forward(x)
        action_log_prob = torch.zeros(action.shape[0], 1).to(device)

        control_action = action
        control_action_log_prob_nodes = control_dist.log_prob(control_action)
        control_action_log_prob_cum = torch.cumsum(control_action_log_prob_nodes, dim=0)
        # control_action_log_prob_cum = control_action_log_prob_cum[torch.LongTensor(num_nodes_cum_control) - 1]
        control_action_log_prob = torch.cat(
            [control_action_log_prob_cum[[0]], control_action_log_prob_cum[1:] - control_action_log_prob_cum[:-1]])
        action_log_prob = control_action_log_prob
        print(action_log_prob)
        return action_log_prob
