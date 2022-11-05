from abc import ABC

from dm_control.mujoco import Physics
from dm_control.rl.control import Environment
from dm_control.suite import base
from dm_control import mujoco
import os


class MujocoEnv(Environment):
    """
        Superclass for all MujoCo environments.
    """

    def __init__(self, cfg, mujoco_xml_string=None, mujoco_xml_path=None, *args, **kwargs):
        self.cfg = cfg
        # create an object Physics
        # if xml content already exits in cfg file, it is preferred to be used.
        # otherwise, load .xml from path.
        if mujoco_xml_string is not None:
            self._physics = Physics.from_xml_string(mujoco_xml_string)
        else:
            if not os.path.exists(mujoco_xml_path):
                raise IOError('XML file %s does not exist!' % mujoco_xml_path)
            else:
                self._physics = Physics.from_xml_path(mujoco_xml_path)
        self._task = MujocoTask(cfg)
        super(MujocoEnv, self).__init__(self._physics, self._task, *args, **kwargs)


class MujocoTask(base.Task):
    def __init__(self, cfg):
        super().__init__()
