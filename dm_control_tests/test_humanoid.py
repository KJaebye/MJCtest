"""
    The render should pop up and the simulation should be running.
    Double-click on a geom and hold Ctrl to apply forces (using right mouse button) and torques (using left mouse button).
"""

from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="humanoid", task_name="stand")
action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  # x = np.random.uniform(low=action_spec.minimum,
  #                       high=action_spec.maximum,
  #                       size=action_spec.shape)
  x = np.random.uniform(low=-3,
                        high=4,
                        size=action_spec.shape)
  print(list(x))
  print(type(x))
  return x

# Launch the viewer application.
viewer.launch(env, policy=random_policy)