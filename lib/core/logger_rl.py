# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class LoggerRL
#   @author: From khrylib by Ye Yuan, modified by Kangyao Huang
#   @changes: Use self.sample_duration to replace self.sample_time
#   @created date: 28.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import math
from lib.core.stats_logger import StatsLogger


class LoggerRL:
    """
        Actually this class is not for logging, but recording the variable values during the training.
    """
    def __init__(self, init_stats_logger=True, use_c_reward=False):
        self.use_c_reward = use_c_reward
        self.num_steps = 0
        self.num_episodes = 0
        self.sample_duration = 0
        self.episode_len = 0
        self.episode_reward = 0
        self.episode_c_reward = 0

        self.stats_names = ['episode_len', 'reward', 'episode_reward']
        self.stats_nparray = []
        if init_stats_logger:
            self.stats_loggers = {x: StatsLogger(is_nparray=x in self.stats_nparray) for x in self.stats_names}

    def start_episode(self, env):
        self.episode_len = 0
        self.episode_reward = 0
        self.episode_c_reward = 0

    def step(self, reward):
        self.episode_len += 1
        self.episode_reward += reward
        self.stats_loggers['reward'].log(reward)

    def end_episode(self):
        self.num_steps += self.episode_len
        self.num_episodes += 1
        self.stats_loggers['episode_len'].log(self.episode_len)
        self.stats_loggers['episode_reward'].log(self.episode_reward)

    def end_sampling(self):
        pass

    @classmethod
    def merge(cls, logger_list, **kwargs):
        logger = cls(init_stats_logger=False, **kwargs)
        logger.num_episodes = sum([x.num_episodes for x in logger_list])
        logger.num_steps = sum([x.num_steps for x in logger_list])
        logger.stats_loggers = {}
        for stats in logger.stats_names:
            logger.stats_loggers[stats] = StatsLogger.merge([x.stats_loggers[stats] for x in logger_list])

        logger.total_reward = logger.stats_loggers['reward'].total()
        logger.avg_episode_len = logger.stats_loggers['episode_len'].avg()
        logger.avg_episode_reward = logger.stats_loggers['episode_reward'].avg()
        logger.max_episode_reward = logger.stats_loggers['episode_reward'].max()
        logger.min_episode_reward = logger.stats_loggers['episode_reward'].min()
        logger.max_reward = logger.stats_loggers['reward'].max()
        logger.min_reward = logger.stats_loggers['reward'].min()
        logger.avg_reward = logger.stats_loggers['reward'].avg()
        if logger.use_c_reward:
            logger.total_c_reward = logger.stats_loggers['c_reward'].total()
            logger.avg_c_reward = logger.stats_loggers['c_reward'].avg()
            logger.max_c_reward = logger.stats_loggers['c_reward'].max()
            logger.min_c_reward = logger.stats_loggers['c_reward'].min()
            logger.total_c_info = logger.stats_loggers['c_info'].total()
            logger.avg_c_info = logger.stats_loggers['c_info'].avg()
            logger.avg_episode_c_reward = logger.stats_loggers['episode_c_reward'].avg()
            logger.avg_episode_c_info = logger.stats_loggers['c_info'].total() / logger.num_episodes
        return logger
