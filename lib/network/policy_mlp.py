# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class PolicyMLP
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 03.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

import torch

from lib.network.policy import Policy
from lib.core.distributions import DiagGaussian


class PolicyMLP(Policy):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = torch.nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(torch.nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = torch.nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)  # no bias?

        self.action_log_std = torch.nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        dist = DiagGaussian(action_mean, action_std)
        return dist
