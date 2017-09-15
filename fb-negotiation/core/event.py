from cocoa.core.event import Event as BaseEvent

class Event(BaseEvent):
    @staticmethod
    def SelectEvent(agent, data, time=None):
        return Event(agent, time, 'select', data)

    @staticmethod
    def RejectEvent(agent, time=None):
        return Event(agent, time, 'reject', None)

