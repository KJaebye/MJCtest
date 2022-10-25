# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Set configurations of training or evaluation.
#   @author: Kangyao Huang
#   @created date: 25.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

"""
    This file is to load all optional settings of scenes and environments from a .yml file.
    Meanwhile, it creates output directories for log files, tensorboard summary, checkpoints and models. Results of
    training are saved at /tmp in default, well-trained results are then moved to /results. Unless, setting '--tmp' to
    False can save results in /results directly.
"""

import glob
import os
import yaml
from datetime import datetime

class Config:
    def __init__(self, domain, tmp, task, cfg_dict=None):
        self.domain = domain
        self.task = task

        # load .yml
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_path = '../config/cfg/**/%s/%s.yml' % (domain, task)
            files = glob.glob(cfg_path, recursive=True)
            assert len(files) == 1
            cfg = yaml.safe_load(open(files[0], 'r'))

        # create directories
        output_dir = '../tmp' if tmp else '../results'
        subdir = '/%s/%s' % (domain, task)
        target_dir = '/' + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir + subdir + target_dir
        self.model_dir = '%s/models' % self.output_dir
        self.log_dir = '%s/log' % self.output_dir
        self.tb_dir = '%s/tb' % self.output_dir
        os.makedirs(self.model_dir, exist_ok=False)
        os.makedirs(self.log_dir, exist_ok=False)
        os.makedirs(self.tb_dir, exist_ok=False)

        # training config

        # env
        self.env_name = cfg.get('env_name')
        self.task_complexity = cfg.get('task_complexity')

        # robot
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())