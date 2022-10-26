# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Logger.
#   @author: Kangyao Huang
#   @created date: 26.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import logging
import sys
import os
from termcolor import colored
import platform

NOTSET = 0
LOGGING_METHOD = ['info', 'warning', 'error', 'critical',
                  'warn', 'exception', 'debug']


class MyFormatter(logging.Formatter):
    """
         A class to make preference format.
    """

    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'

        if record.levelno == logging.WARNING:
            fmt = date + ' ' + \
                  colored('WRN', 'red', attrs=[]) + ' ' + msg
        elif record.levelno == logging.ERROR or \
                record.levelno == logging.CRITICAL:
            fmt = date + ' ' + \
                  colored('ERR', 'red', attrs=['underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg

        if hasattr(self, '_style'):
            # Python3 compatibilty
            self._style._fmt = fmt
        self._fmt = fmt

        return super(self.__class__, self).format(record)


class Logger(logging.Logger):

    def __init__(self, name, args=None, cfg=None, level=NOTSET):
        super(Logger, self).__init__(name, level)
        self.file_path = None
        self.args = args
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.time_str = cfg.time_str
        self.prefix = cfg.domain + '_' + cfg.task + '-'
        self.file_name = self.prefix + cfg.time_str + '.log'

    def print_system_info(self):
        # print necessary info
        self.info('Hardware info: {}'.format(platform.processor()))
        self.info('System info: {}'.format(platform.system()))
        self.info('Current Python version: {}'.format(platform.python_version()))
        return

    def set_output_handler(self):
        # set the console output handler
        con_handler = logging.StreamHandler(sys.stdout)
        con_handler.setFormatter(MyFormatter(datefmt='%Y%m%d %H:%M:%S'))
        self.addHandler(con_handler)
        return

    def set_file_handler(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.file_path = os.path.join(self.log_dir, self.file_name)
        file_handler = logging.FileHandler(
            filename=self.file_path, encoding='utf-8', mode='w')
        file_handler.setFormatter(MyFormatter(datefmt='%Y%m%d %H:%M:%S'))
        self.addHandler(file_handler)
        self.info('Log file set to {}'.format(self.file_path))
        return
