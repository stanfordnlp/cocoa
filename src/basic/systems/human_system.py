__author__ = 'anushabala'
from system import System
from src.basic.sessions.human_session import HumanSession


class HumanSystem(System):
    def __init__(self):
        super(HumanSystem, self).__init__()

    @classmethod
    def name(cls):
        return 'human'

    def new_session(self, agent, kb):
        return HumanSession(agent)
