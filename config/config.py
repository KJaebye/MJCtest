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
import numpy as np


class Config:
    def __init__(self, domain, task, rec=None):
        self.domain = domain
        self.task = task
        self.rec = rec

        # load .yml
        cfg_path = './config/cfg/**/%s/%s.yml' % (domain, task)
        files = glob.glob(cfg_path, recursive=True)
        assert len(files) == 1, "{} file(s) is/are found.".format(len(files))
        cfg = yaml.safe_load(open(files[0], 'r'))

        # training config
        self.env_name = cfg.get('env_name')
        self.agent_spec = cfg.get('agent_spec', dict)
        self.gamma = cfg.get('gamma', 0.995)
        self.tau = cfg.get('tau', 0.95)

        self.policy_spec = cfg.get('policy_spec', dict())
        self.policy_optimizer = cfg.get('policy_optimizer', 'Adam')
        self.policy_lr = cfg.get('policy_lr', 5e-5)
        self.policy_momentum = cfg.get('policy_momentum', 0.0)
        self.policy_weight_decay = cfg.get('policy_weight_decay', 0.0)

        self.value_spec = cfg.get('value_spec', dict())
        self.value_optimizer = cfg.get('value_optimizer', 'Adam')
        self.value_lr = cfg.get('value_lr', 3e-4)
        self.value_momentum = cfg.get('value_momentum', 0.0)
        self.value_weight_decay = cfg.get('value_weight_decay', 0.0)

        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.l2_reg = cfg.get('l2_reg', 1e-3)
        self.entropy_coeff = cfg.get('entropy_coeff', 1.e-2)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.optim_num_epoch = cfg.get('optim_num_epoch', 10)
        self.batch_size = cfg.get('batch_size', 2048)
        self.eval_batch_size = cfg.get('eval_batch_size', 2048)
        self.mini_batch_size = cfg.get('mini_batch_size', 64)
        self.max_iter_num = cfg.get('max_iter_num', 1000)
        self.save_model_interval = cfg.get('save_model_interval', 100)
        self.seed = cfg.get('seed', 1)

        # robot
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())

