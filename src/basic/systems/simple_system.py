__author__ = 'anushabala'
from system import System
from src.basic.sessions.simple_session import SimpleSession
from src.basic.sessions.timed_session import TimedSessionWrapper


class SimpleSystem(System):
    def __init__(self, lexicon, timed_session=False):
        super(SimpleSystem, self).__init__()
        self.lexicon = lexicon
        self.timed_session = timed_session

    @classmethod
    def name(cls):
        return 'simple'

    def new_session(self, agent, kb):
        session = SimpleSession(agent, kb, self.lexicon)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
	return session
