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


class CentipedeEnv(MujocoEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mujoco_xml_string = None
        self.mujoco_xml_path = "./assets/robot_models/centipede/centipede_four.xml"
        physics = CentipedePhysics.from_xml_path(self.mujoco_xml_path)
        task = CentipedeTask()
        super().__init__(physics, task)

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass



class CentipedePhysics(MujocoPhysics):
    pass


class CentipedeTask(MujocoTask):
    def __init__(self):
        super().__init__()
