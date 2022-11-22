# ------------------------------------------------------------------------------------------------------------------- #
#   @description: common functions
#   @author: From khrylib by Ye Yuan, by Kangyao Huang
#   @created date: 13.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.core import torch_wrapper as torper
import torch


def estimate_advantages(rewards, values, gamma, tau):
    device = rewards.device
    rewards, values = torper.batch_to(torch.device('cpu'), rewards, values)
    tensor_type = type(rewards)
    delta = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        delta[i] = rewards[i] + gamma * prev_value - values[i]
        advantages[i] = delta[i] + gamma * tau * prev_advantage

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]


    returns = values + advantages
    # normalization
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = torper.batch_to(device, advantages, returns)
    return advantages, returns
