__author__ = 'anushabala'

from src.basic.systems.system import System


class NgramSystem(object):
    def __init__(self, data_path, n=5):
        super(NgramSystem, self).__init__()
        self.data_path = data_path
        self.n = n