from cocoa.sessions.session import Session as BaseSession
from core.event import Event

class Session(BaseSession):
    def offer(self, offer, metadata=None):
        """Send an offer event.

        Args:
            offer ({'price': float, 'sides', str})

        Returns:
            OfferEvent

        """
        return Event.OfferEvent(self.agent, offer, time=self.timestamp(), metadata=None)

    def accept(self, metadata=None):
        return Event.AcceptEvent(self.agent, time=self.timestamp(), metadata=None)

    def reject(self, metadata=None):
        return Event.RejectEvent(self.agent, time=self.timestamp(), metadata=None)

    def quit(self, metadata=None):
        return Event.QuitEvent(self.agent, time=self.timestamp(), metadata=None)
