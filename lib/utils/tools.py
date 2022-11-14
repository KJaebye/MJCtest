# ------------------------------------------------------------------------------------------------------------------- #
#   @description: tool functions
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 14.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

def get_state(observation):
    state = []
    for s in observation.values():
        state += list(s)
    return state


def init_fc_weights(fc):
    fc.weight.data.mul_(0.1)
    fc.bias.data.mul_(0.0)

def index_select_list(x, ind):
    return [x[i] for i in ind]