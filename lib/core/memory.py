# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Memory (Reply Buffer).
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 27.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import random


class Memory:
    """ Reply Buffer. """
    def __int__(self):
        self.memory = []

    def push(self, *args):
        """ Saves a tuple. """
        self.memory.append([*args])

    def sample(self, batch_size=None):
        return random.sample(self.memory, batch_size) if batch_size is not None else self.memory

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)