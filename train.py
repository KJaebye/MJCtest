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
from structural_control.agents.hopper_agent import HopperAgent
from structural_control.envs.hopper import HopperEnv

if __name__ == "__main__":
    args = get_args()

    if args.render:
        args.num_threads = 1

    """load env configs and training settings"""
    cfg = Config(args.domain, args.task, tmp=True, cfg_dict=None)

    """set torch and cuda"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) \
        if args.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.is_available() is natively False on mac m1
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    """logging config"""
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

    start_epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch

    """create agent"""
    agent = HopperAgent(cfg, logger, dtype=dtype, device=device, seed=cfg.seed, num_threads=args.num_threads,
                        training=True, checkpoint=start_epoch)

    if args.render:
        agent.pre_epoch_update(start_epoch)
        agent.sample(1e8, mean_action=not args.show_noise, render=True)
    else:
        for epoch in range(start_epoch, cfg.max_epoch_num):
            agent.optimize(epoch)
            # clean up GPU memory
            torch.cuda.empty_cache()
        agent.logger.critical('Training completed!')
