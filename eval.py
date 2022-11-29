# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Evaluation file
#   @author: Kangyao Huang
#   @created date: 17.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

import logging
import torch
import numpy as np
import argparse

from config.config import Config
from utils.logger import Logger
from structural_control.agents.pendulum_agent import PendulumAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='hopper', help='mujoco domain')
    parser.add_argument('--task', type=str, default='easy', help='task complexity')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--epoch', default='best')
    parser.add_argument('--save_video', action='store_true', default=False)

    args = parser.parse_args()

    """ load env configs and training settings """
    cfg = Config(args.domain, args.task, tmp=True, cfg_dict=None)

    """ set torch and cuda """
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    """ logging config """
    # set logger
    logger = Logger(name='current', args=args, cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # set output
    logger.set_output_handler()
    logger.print_system_info()

    # only training generates log file
    logger.critical('Type of current running: Evaluation. No log file will be created')
    logger.set_file_handler()

    # iter = 'best'
    iter = 5900

    """ create agent """
    # agent = HopperAgent(cfg, logger, dtype=dtype, device=device, seed=cfg.seed, num_threads=1,
    #                     render=True, training=False, checkpoint=epoch)
    agent = PendulumAgent(cfg, logger, dtype=dtype, device=device, num_threads=1, training=False, checkpoint=iter)

    agent.visualize_agent(num_episode=1, save_video=args.save_video)

