from src.basic.event import BaseEvent

class Event(BaseEvent):
    @staticmethod
    def OfferEvent(agent, data, time=None):
        return Event(agent, time, 'offer', data)

    @staticmethod
    def QuitEvent(agent, data, time=None):
        return Event(agent, time, 'quit', data)
