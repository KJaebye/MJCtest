# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class LoggerRL
#   @author: From khrylib by Ye Yuan, modified by Kangyao Huang
#   @changes: Use self.sample_duration to replace self.sample_time
#   @created date: 28.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #


class LoggerRL:
    """
        Actually this class is not for logging, but record the variable values during the training.
    """
    def __init__(self):
        self.num_steps = 0
        self.num_episodes = 0
        self.episode_len = 0
        self.sample_duration = 0

        self.episode_reward = 0
        self.total_reward = 0
        self.avg_episode_reward = 0
        self.min_episode_reward = 1e6
        self.max_episode_reward = -1e6

        self.episode_custom_reward = 0
        self.total_custom_reward = 0
        self.avg_custom_reward = 0
        self.min_episode_custom_reward = 1e6
        self.max_episode_custom_reward = -1e6

    def start_episode(self):
        self.episode_len = 0

        self.episode_reward = 0
        self.episode_custom_reward = 0

    def step(self, reward):
        self.episode_len += 1
        self.episode_reward += reward

    def step_custom(self, reward):
        self.total_custom_reward += reward

    def end_episode(self):
        self.num_steps += self.episode_len
        self.num_episodes += 1
        # reward
        self.total_reward += self.episode_reward
        self.min_episode_reward = min(self.min_episode_reward, self.episode_reward)
        self.max_episode_reward = max(self.max_episode_reward, self.episode_reward)
        # custom reward
        self.total_custom_reward += self.episode_custom_reward
        self.min_episode_custom_reward = min(self.min_episode_custom_reward, self.episode_custom_reward)
        self.max_episode_custom_reward = max(self.max_episode_custom_reward, self.episode_custom_reward)

    def end_sampling(self):
        self.avg_episode_reward = self.total_reward / self.num_episodes
        self.avg_custom_reward = self.total_custom_reward / self.num_episodes


    @classmethod
    def merge(cls, logger_list, use_custom_reward=False):
        logger = cls()
        logger.num_episodes = sum([x.num_episodes for x in logger_list])
        logger.num_steps = sum([x.num_steps for x in logger_list])
        logger.avg_episode_len = logger.num_steps / logger.num_episodes

        logger.total_reward = sum([x.total_reward for x in logger_list])
        logger.avg_episode_reward = logger.total_reward / logger.num_episodes
        logger.max_episode_reward = max([x.max_episode_reward for x in logger_list])
        logger.min_episode_reward = min([x.max_episode_reward for x in logger_list])

        if use_custom_reward:
            logger.total_custom_reward = sum([x.total_custom_reward for x in logger_list])
            logger.avg_episode_custom_reward = logger.total_custom_reward / logger.num_episodes
            logger.max_episode_custom_reward = max([x.max_episode_custom_reward for x in logger_list])
            logger.min_episode_custom_reward = min([x.max_episode_custom_reward for x in logger_list])
        return logger