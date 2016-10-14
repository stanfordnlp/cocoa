__author__ = 'anushabala'
from system import System
from src.basic.sessions.simple_session import SimpleSession


class SimpleSystem(System):
    def __init__(self):
        super(SimpleSystem, self).__init__()
        # todo do we need to do anything here?

    @classmethod
    def name(cls):
        return 'simple'

    def new_session(self, agent, kb):
        return SimpleSession(agent, kb)