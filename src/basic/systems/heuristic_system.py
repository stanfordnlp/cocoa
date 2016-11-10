__author__ = 'anushabala'
from system import System
from src.basic.sessions.heuristic_session import HeuristicSession


class HeuristicSystem(System):
    def __init__(self):
        super(HeuristicSystem, self).__init__()
        # todo do we need to do anything here?

    @classmethod
    def name(cls):
        return 'heuristic'

    def new_session(self, agent, kb):
        return HeuristicSession(agent, kb)
