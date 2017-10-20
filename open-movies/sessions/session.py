from cocoa.sessions.session import Session as BaseSession
from core.event import Event

class Session(BaseSession):
    # def select(self, outcome):
    #     return Event.SelectEvent(self.agent, data=outcome, time=self.timestamp())

    def done(self):
        return Event.DoneEvent(self.agent, time=self.timestamp())
