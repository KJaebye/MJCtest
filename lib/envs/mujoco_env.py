# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class MujocoEnv
#   @author: by Kangyao Huang
#   @created date: 05.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
from abc import ABC

from dm_control.rl.control import Environment
from dm_control.mujoco import Physics
from dm_control.suite.base import Task


class MujocoEnv(Environment):
    """
        Superclass for all MujoCo environments in this proj.
        This class will pass config into environment.
    """

    def __init__(self, cfg, physics, task, *args, **kwargs):
        """
        :param physics: Instance of Physics.
        :param task: Instance of Task.
        :param args: ...
        :param kwargs: ...
        """
        self.cfg = cfg
        super(MujocoEnv, self).__init__(physics, task, *args, **kwargs)

    def seed(self, seed=None):
        self.np_random, seed = seeding

class MujocoPhysics(Physics):
    pass


class MujocoTask(Task, ABC):
    def __init__(self, random=None):
        super().__init__(random)
