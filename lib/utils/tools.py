def get_state(observation):
    state = []
    for s in observation.values():
        state += list(s)
    return state


def init_fc_weights(fc):
    fc.weight.data.mul_(0.1)
    fc.bias.data.mul_(0.0)
