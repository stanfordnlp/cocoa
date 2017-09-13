from cocoa.core.event import Event as BaseEvent

class Event(BaseEvent):
    @staticmethod
    def SelectEvent(agent, data, time=None):
        return Event(agent, time, 'select', data)

