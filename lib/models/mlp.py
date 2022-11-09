# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class MLP
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 03.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.output_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(torch.nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x
