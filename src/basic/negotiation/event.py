from src.basic.event import BaseEvent

class Event(BaseEvent):
    @staticmethod
    def OfferEvent(agent, data, time=None):
        return Event(agent, time, 'offer', data)

    @staticmethod
    def QuitEvent(agent, data, time=None):
        return Event(agent, time, 'quit', data)

    @staticmethod
    def AcceptEvent(agent, time=None):
        return Event(agent, time, 'accept', None)

    @staticmethod
    def RejectEvent(agent, time=None):
        return Event(agent, time, 'reject', None)
