# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class CentipedeEnv, CentipedePhysics, CentipedeTask
#   @author: by Kangyao Huang
#   @created date: 05.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    When creating an environment for RL-MuJoCo, three classes should be defined.
    Use centipede as an example:
    1. Class CentipedeEnv: inherit from MujocoEnv
    2. Class CentipedePhysics: inherit from MujocoPhysics
    3. Class CentipedeTask: inherit from MujocoTask
    Instantiate CentipedeEnv will also instantiate CentipedePhysics and
    CentipedeTask. Besides, Instantiated Physics and Task will be passed as
    parameters to lower level wrapper Environment located at dm_control.rl.control
"""
from abc import ABC

from lib.envs.mujoco_env import MujocoEnv, MujocoPhysics, MujocoTask
from dm_control.rl.control import flatten_observation
import dm_env


class CentipedeEnv(MujocoEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mujoco_xml_string = None
        self.mujoco_xml_path = "./assets/robot_models/centipede/centipede_four.xml"
        self._physics = CentipedePhysics.from_xml_path(self.mujoco_xml_path)
        self._task = CentipedeTask()
        super().__init__(self._physics, self._task, flat_observation=False)
        self._step_count = 0

    def reset(self):
        """ Starts a new episode and returns the first `TimeStep`. """
        self._reset_next_step = False
        self._step_count = 0

        # initiate states/configuration of physics
        with self._physics.reset_context():
            self._task.initialize_episode(self._physics)

        # get initial observation
        observation = self._task.get_observation(self._physics)
        if self._flat_observation:
            observation = flatten_observation(observation)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)

    def step(self, action):
        pass

    def render(self):
        pass


class CentipedePhysics(MujocoPhysics):
    pass


class CentipedeTask(MujocoTask):
    def __init__(self, random=None):
        """ Initialize an instance of `Hopper`."""
        super().__init__(random=None)

    def initialize_episode(self, physics):


        super().initialize_episode(physics)

    def get_observation(self, physics):
        pass

    def get_reward(self, physics):
        pass
