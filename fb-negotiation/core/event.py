from cocoa.core.event import Event as BaseEvent

class Event(BaseEvent):
    @staticmethod
    def SelectEvent(agent, data, time=None, metadata=None):
        return Event(agent, time, 'select', data, metadata=metadata)

    @staticmethod
    def RejectEvent(agent, time=None, metadata=None):
        return Event(agent, time, 'reject', None, metadata=metadata)

