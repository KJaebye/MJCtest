# ------------------------------------------------------------------------------------------------------------------- #
#   @description: This file parses running arguments from terminal.
#   @author: Kangyao Huang
#   @created date: 24.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import argparse


def get_args():
    # create a parser
    parser = argparse.ArgumentParser(description="Write in user's arguments from terminal.")

    # settings
    parser.add_argument('--domain', type=str, help='mujoco domain, must be specified to load the cfg file.', required=True)
    parser.add_argument('--task', type=str, help='task, must be specified to load the cfg file.', required=True)
    parser.add_argument('--algo', type=str, default='PPO2', help='algorithm to train the agent')
    parser.add_argument('--use_cuda', type=bool, default=False)

    # training configuration
    parser.add_argument('--rec', type=str)
    parser.add_argument('--start_iter', default='0')
    parser.add_argument('--num_threads', type=int, default=1)

    # settings for networks
    parser.add_argument('--use_ggnn', type=bool, default=False, help='use NerveNet(GGNN) as policy networks')

    # parsing and return args
    return parser.parse_args()


if __name__ == '__main__':
    pass
