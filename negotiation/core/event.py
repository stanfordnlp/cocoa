from cocoa.core.event import Event as BaseEvent

class Event(BaseEvent):
    @staticmethod
    def OfferEvent(agent, data, time=None, metadata=None):
        return Event(agent, time, 'offer', data, metadata=metadata)

    @staticmethod
    def QuitEvent(agent, time=None, metadata=None):
        return Event(agent, time, 'quit', None, metadata=metadata)

    @staticmethod
    def AcceptEvent(agent, time=None, metadata=None):
        return Event(agent, time, 'accept', None, metadata=metadata)

    @staticmethod
    def RejectEvent(agent, time=None, metadata=None):
        return Event(agent, time, 'reject', None, metadata=metadata)
