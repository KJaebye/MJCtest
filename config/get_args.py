# ------------------------------------------------------------------------------------------------------------------- #
#   @description: This file parses running arguments from terminal.
#   @author: Kangyao Huang
#   @created date: 24.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import argparse


def get_args():
    # create a parser
    parser = argparse.ArgumentParser(description="Write in user's arguments from terminal.")

    # the experiment settings
    parser.add_argument('--type', type=str, default='training', help='training or evaluation')
    parser.add_argument('--domain', type=str, default='domain', help='mujoco domain')
    parser.add_argument('--task', type=str, default='easy', help='task complexity')
    parser.add_argument('--algo', type=str, default='PPO', help='algorithm to train the agent')
    parser.add_argument('--tmp', type=bool, default=True)
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)

    # training configuration
    parser.add_argument('--gamma', type=float, default=.99, help='discount factor for value function')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes')
    parser.add_argument('--timesteps_per_episode', type=int, default=2050, help='number of steps per rollout(episode)')

    # settings for networks
    parser.add_argument('--use_ggnn', type=bool, default=False, help='use NerveNet(GGNN) as policy networks')

    # parsing and return args
    return parser.parse_args()


if __name__ == '__main__':
    pass
