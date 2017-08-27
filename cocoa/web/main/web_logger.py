__author__ = 'anushabala'

import logging
from logging import Logger

logger = None


class WebLogger(object):
    @classmethod
    def initialize(cls, log_file):
        logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
        global logger
        logger = logging.getLogger("web")

    @classmethod
    def get_logger(cls):
        global logger
        if not logger:
            logging.basicConfig(filename="web.log", filemode='w')
            logger = logging.getLogger("web")
        return logger
