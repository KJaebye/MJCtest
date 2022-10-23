# --------------------------------------------------------------------------------
#   @description: This file parses running arguments from terminal.
#   @author: Modified from Github:WilsonWangTHU/NerveNet, by Kangyao Huang
#   @create date: 24.Oct.2022
# --------------------------------------------------------------------------------

import argparse

def parse_args():
    # create a parser
    parser = argparse.ArgumentParser(description="Write in user's arguments from terminal.")

    # the experiment settings
    parser.add_argument("--task", type=str, default='centipede', help='the mujoco environment to test')
    parser.add_argument("--output_dir", type=str, default='../checkpoint/')

    # training configuration
    parser.add_argument("--gamma", type=float, default=.99, help='the discount factor for value function')

    # parsing and return args
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pass