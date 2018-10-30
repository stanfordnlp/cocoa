from system import System
from cocoa.sessions.timed_session import TimedSessionWrapper

class RulebasedSystem(System):
    def __init__(self, lexicon, generator, manager, timed_session):
        super(RulebasedSystem, self).__init__()
        self.timed_session = timed_session
        self.lexicon = lexicon
        self.generator = generator
        self.manager = manager

    @classmethod
    def name(cls):
        return 'rulebased'

    def new_session(self, agent, kb, config=None):
        session = self._new_session(agent, kb, config)
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session

    def _new_session(self, agent, kb, config=None):
        raise NotImplementedError
