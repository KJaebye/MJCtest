


def get_state(observation):
    state = []
    for s in observation.values():
        state += list(s)
    return state