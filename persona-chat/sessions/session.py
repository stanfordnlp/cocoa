from cocoa.sessions.session import Session as BaseSession
from core.event import Event

class Session(BaseSession):
    def done(self):
        return Event.DoneEvent(self.agent, time=self.timestamp())