# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class CentipedeEnv
#   @author: by Kangyao Huang
#   @created date: 04.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from dm_control.rl.control import Physics
from dm_control.rl.control import Task
from dm_control.mujoco.engine import Physics

class Centipede(Physics):
    def __int__(self, cfg):
        self.cfg = cfg
        self.physics_path = cfg.physics_path
        self.load_xml()

    def load_xml(self):
        self.from_xml_path(self.physics_path)




