from cocoa.systems.system import System

from sessions.rl_session import RLSession

class RLSystem(System):
    def __init__(self, system):
        self.system = system
        # TODO: add optimizer here

    @classmethod
    def name(cls):
        return 'RL-{}'.format(self.system.name())

    def new_session(self, agent, kb):
        session = self.system.new_session(agent, kb)
        rl_session = RLSession(session, self.opt)
        return rl_session

