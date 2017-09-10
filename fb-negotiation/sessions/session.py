from cocoa.sessions.session import Session as BaseSession
from core.event import Event

class Session(BaseSession):
    def mark_deal_agreed(self, data):
        return Event.SelectEvent(self.agent, time=self.timestamp(), data=data)

    # def accept(self, data):
    #     return Event.AcceptEvent(self.agent, time=self.timestamp())

    # def reject(self):
    #     return Event.RejectEvent(self.agent, time=self.timestamp())

    # def quit(self):
    #     return Event.QuitEvent(self.agent, time=self.timestamp())
