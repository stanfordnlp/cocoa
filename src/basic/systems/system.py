__author__ = 'anushabala'


class System(object):
    def __init__(self):
        pass

    def new_session(self, agent, kb):
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError
