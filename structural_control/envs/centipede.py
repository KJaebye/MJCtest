# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class CentipedeEnv
#   @author: by Kangyao Huang
#   @created date: 05.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from lib.envs.mujoco_env import MujocoEnv


class CentipedeEnv(MujocoEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mujoco_xml_string = None
        self.mujoco_xml_path = "./assets/robot_models/centipede/centipede_four.xml"
        super().__init__(self.cfg, mujoco_xml_string=self.mujoco_xml_string, mujoco_xml_path=self.mujoco_xml_path)
