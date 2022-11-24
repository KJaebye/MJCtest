import sys
sys.path.append('../')

from structural_control.envs.pendulum import PendulumEnv
from dm_control import viewer
import numpy as np



env = PendulumEnv(None)
action_spec = env.action_spec()

def random_policy(time_step):
    del time_step  # Unused.
    x = np.random.uniform(low=action_spec.maximum,
                        high=action_spec.maximum,
                        size=action_spec.shape)
    # print(x.shape)
    # print(x)
    # print(type(x))
    return x


viewer.launch(env, policy=random_policy)