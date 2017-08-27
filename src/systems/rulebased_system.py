from system import BaseSystem
from src.sessions.timed_session import TimedSessionWrapper

class BaseRulebasedSystem(BaseSystem):
    def __init__(self, timed_session):
        super(BaseRulebasedSystem, self).__init__()
        self.timed_session = timed_session

    @classmethod
    def name(cls):
        return 'rulebased'

    def new_session(self, agent, kb):
        session = self._new_session(agent, kb)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
        return session

    def _new_session(self, agent, kb):
        raise NotImplementedError
