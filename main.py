# ------------------------------------------------------------------------------------------------------------------- #
#   @description: The main file of training an agent.
#   @author: Kangyao Huang
#   @created date: 23.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #
import logging
from config.get_args import get_args
from config.config import Config
from utils.logger import Logger


if __name__ == "__main__":
    args = get_args()

    if args.render:
        args.num_threads = 1

    """load env configs and training settings"""
    cfg = Config(args.domain, args.task, tmp=True, cfg_dict=None)

    # """set torch and cuda"""
    # dtype = torch.float64
    # torch.set_default_dtype(dtype)
    # device = torch.device('cuda', index=args.gpu_index) \
    #     if args.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    # # torch.cuda.is_available() is natively False on mac m1
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(args.gpu_index)
    # np.random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)

    """logging config"""
    # set logger
    logger = Logger(name='current', args=args, cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # set output and logfile
    logger.set_output_handler()
    logger.set_file_handler()
    # print info
    logger.print_system_info()

    # if args.train:

    """create agent"""



