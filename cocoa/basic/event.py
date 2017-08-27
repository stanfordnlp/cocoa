class BaseEvent(object):
    """
    An atomic event of a dialogue, which could be someone talking or making a selection.

    Params:
    agent: The index of the agent triggering the event
    time: Time at which event occurred
    action: The action this event corresponds to ('select', 'message', ..)
    data: Any data that is part of the event
    start_time: The time at which the event action was started (e.g. the time at which an agent starting typing a
    message to send)
    """
    def __init__(self, agent, time, action, data, start_time=None):
        self.agent = agent
        self.time = time
        self.action = action
        self.data = data
        self.start_time = start_time

    @staticmethod
    def from_dict(raw):
        return Event(raw['agent'], raw['time'], raw['action'], raw['data'], start_time=raw.get('start_time'))

    def to_dict(self):
        return {'agent': self.agent, 'time': self.time, 'action': self.action, 'data': self.data,
                'start_time': self.start_time}

    @staticmethod
    def MessageEvent(agent, data, time=None, start_time=None):
        return Event(agent, time, 'message', data, start_time=start_time)

    @staticmethod
    def JoinEvent(agent, userid=None, time=None):
        return Event(agent, time, 'join', userid)

    @staticmethod
    def LeaveEvent(agent, userid=None, time=None):
        return Event(agent, time, 'leave', userid)

    @staticmethod
    def TypingEvent(agent, data, time=None):
        return Event(agent, time, 'typing', data)

    @staticmethod
    def EvalEvent(agent, data, time):
        return Event(agent, time, 'eval', data)

    @classmethod
    def gather_eval(cls, events):
        event_dict = {e.time: e for e in events if e.action != 'eval'}
        for e in events:
            if e.action == 'eval':
                event_dict[e.time].tags = [k for k, v in e.data['labels'].iteritems() if v != 0]
            else:
                event_dict[e.time].tags = []
        events_with_eval = [v for k, v in sorted(event_dict.iteritems(), key=lambda x: x[0])]
        return events_with_eval


import src.config as config
import importlib
task_module = importlib.import_module('.'.join(('src.basic', config.task, 'event')))
Event = task_module.Event
