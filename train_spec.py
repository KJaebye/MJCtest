# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Training file
#   @author: Kangyao Huang
#   @created date: 23.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import logging
import torch
import numpy as np
from config.get_args import get_args
from config.config import Config
from utils.logger import Logger
from structural_control.agents.pendulum_agent import PendulumAgent
from structural_control.agents.walker_agent import WalkerAgent


if __name__ == "__main__":
    args = get_args()
    """ load env configs and training settings """
    cfg = Config(args.domain, args.task)
    """ set torch and cuda """
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) \
        if args.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.is_available() is natively False on mac m1
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
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
    logger.critical('Type of current running: Training')
    logger.set_file_handler()

    start_iter = int(args.start_iter) if args.start_iter.isnumeric() else args.start_iter

    """ create agent """
    if args.domain == 'pendulum':
        agent = PendulumAgent(cfg, logger, dtype=dtype, device=device,
                              num_threads=args.num_threads, training=True, checkpoint=start_iter)
    elif args.domain == 'walker':
        agent = WalkerAgent(cfg, logger, dtype=dtype, device=device,
                            num_threads=args.num_threads, training=True, checkpoint=start_iter)
    else:
        pass

    for iter in range(start_iter, cfg.max_iter_num):
        agent.optimize(iter)
        # clean up GPU memory
        torch.cuda.empty_cache()
    agent.logger.critical('Training completed!')
