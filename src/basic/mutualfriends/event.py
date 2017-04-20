from src.basic.event import BaseEvent

class Event(BaseEvent):
    @staticmethod
    def SelectionEvent(agent, data, time=None):
        return Event(agent, time, 'select', data)

