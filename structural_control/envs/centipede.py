# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class CentipedeEnv, CentipedePhysics, CentipedeTask
#   @author: by Kangyao Huang
#   @created date: 05.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    When creating an environment for MuJoCo, three classes should be defined.
    Use centipede as an example:
    1. Class CentipedeEnv: inherit from MujocoEnv
    2. Class CentipedePhysics: inherit from MujocoPhysics
    3. Class CentipedeTask: inherit from MujocoTask
    Instantiate CentipedeEnv will also instantiate CentipedePhysics and
    CentipedeTask. Besides, Instantiated Physics and Task will be passed as
    parameters to lower level wrapper Environment located at dm_control.rl.control
"""

from lib.envs.mujoco_env import MujocoEnv, MujocoPhysics, MujocoTask


class CentipedeEnv(MujocoEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mujoco_xml_string = None
        self.mujoco_xml_path = "./assets/robot_models/centipede/centipede_four.xml"
        super().__init__(self.cfg, mujoco_xml_string=self.mujoco_xml_string, mujoco_xml_path=self.mujoco_xml_path)
