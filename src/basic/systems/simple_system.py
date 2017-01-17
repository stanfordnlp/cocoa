__author__ = 'anushabala'
from system import System
from src.basic.sessions.simple_session import SimpleSession
from src.basic.sessions.timed_session import TimedSessionWrapper


class SimpleSystem(System):
    def __init__(self, lexicon, timed_session=False, consecutive_entity=True, realizer=None):
        super(SimpleSystem, self).__init__()
        self.lexicon = lexicon
        self.timed_session = timed_session
        self.consecutive_entity = consecutive_entity
        self.realizer = realizer

    @classmethod
    def name(cls):
        return 'simple'

    def new_session(self, agent, kb):
        session = SimpleSession(agent, kb, self.lexicon, self.realizer, self.consecutive_entity)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
	return session
