# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class TrajBatch
#   @author: From khrylib by Ye Yuan, modified by Kangyao Huang
#   @changes: Update the file name to traj_batch.py
#   @created date: 29.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np


class TrajBatch:

    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.cur_states = np.stack(next(self.batch))
        self.actions = np.stack(next(self.batch))
        self.next_states = np.stack(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.masks = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))

