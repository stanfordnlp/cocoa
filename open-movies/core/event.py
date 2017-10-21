from cocoa.core.event import Event as BaseEvent

class Event(BaseEvent):
    @staticmethod
    def DoneEvent(agent, time=None):
        return Event(agent, time, 'done', None)

