"""
    The render should pop up and the simulation should be running.
    Double-click on a geom and hold Ctrl to apply forces (using right mouse button) and torques (using left mouse button).
"""
import numpy as np

from dm_control import viewer
import mujoco_env_use_class as mujoco_env

class HopperEnv(mujoco_env.MujocoEnv):
    def __init__(self, cfg):
        self.mujoco_xml_path = 'hopper.xml'
        import hopper
        Physics_cls = hopper.Physics
        Task_cls = hopper.Hopper
        super().__init__(cfg, Physics_cls, Task_cls, mujoco_xml_path=self.mujoco_xml_path)


env = HopperEnv(None)
# env = suite.hopper.hop()
spec = env.action_spec()


def random_policy(time_step):
    return np.random.uniform(spec.minimum, spec.maximum, spec.shape)


viewer.launch(env, policy=random_policy)


