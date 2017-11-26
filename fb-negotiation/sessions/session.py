from cocoa.sessions.session import Session as BaseSession
from core.event import Event

class Session(BaseSession):
    def select(self, outcome, metadata=None):
        return Event.SelectEvent(self.agent, data=outcome, time=self.timestamp(), metadata=metadata)

    def reject(self, metadata=None):
        return Event.RejectEvent(self.agent, time=self.timestamp(), metadata=metadata)
