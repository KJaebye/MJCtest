# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Value, templet for any critic network
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 12.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

import torch

class Value(torch.nn.Module):
    def __init__(self, net, net_output_dim=None):
        super().__init__()
        self.net = net
        if net_output_dim is None:
            net_output_dim = net.output_dim
        self.value_head = torch.nn.Linear(net_output_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        value = self.value_head(x)
        return value

