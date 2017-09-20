from cocoa.sessions.session import Session as BaseSession
from core.event import Event

class Session(BaseSession):
    def select(self, outcome):
        return Event.SelectEvent(self.agent, time=self.timestamp(), data=outcome)

    def reject(self):
        return Event.RejectEvent(self.agent, time=self.timestamp())

    # def accept(self):
    #     return Event.AcceptEvent(self.agent, time=self.timestamp())

    # def quit(self):
    #     return Event.QuitEvent(self.agent, time=self.timestamp())
