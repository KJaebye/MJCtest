# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Set configurations of training or evaluation
#   @author: Kangyao Huang
#   @created date: 25.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

"""
    This file is to load all optional settings of scenes and environments from a .yml file.
    Meanwhile, it creates output directories for log files, tensorboard summary, checkpoints and models.
    Output files structure (centipede_four as example) likes this:
    /tmp(results)
        /centipede_four
            /easy
                /20221025_235032
                    /model
                    /log
                    /tb
            /hard
                /20221025_235548
                    /model
                    /log
                    /tb
    Results of the training are saved at /tmp in default, well-trained results are then moved to /results. Unless,
    setting '--tmp' to False can save results in /results directly.
"""

import glob
import os
import yaml


class Config:
    def __init__(self, domain, task, tmp=True, cfg_dict=None):
        self.domain = domain
        self.task = task
        self.tmp = tmp

        # load .yml
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_path = './config/cfg/**/%s/%s.yml' % (domain, task)
            files = glob.glob(cfg_path, recursive=True)
            assert len(files) == 1, "{} file(s) is/are found.".format(len(files))
            cfg = yaml.safe_load(open(files[0], 'r'))

        # training config
        self.seed = cfg.get('seed')
        self.min_batch_size = cfg.get('min_batch_size')
        self.max_timesteps = cfg.get('max_timesteps')  # maximum timestep per episode

        # environment



        # robot
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())
