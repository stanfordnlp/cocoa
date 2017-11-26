class Event(object):
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

    decorative_events = ('join', 'leave', 'typing', 'eval')

    def __init__(self, agent, time, action, data, start_time=None, metadata=None):
        self.agent = agent
        self.time = time
        self.action = action
        self.data = data
        self.start_time = start_time
        self.metadata = metadata

    @staticmethod
    def from_dict(raw):
        return Event(raw['agent'], raw['time'], raw['action'], raw['data'], start_time=raw.get('start_time'), metadata=raw.get('metadata'))

    def to_dict(self):
        return {'agent': self.agent, 'time': self.time, 'action': self.action, 'data': self.data,
                'start_time': self.start_time, 'metadata': self.metadata}

    @classmethod
    def MessageEvent(cls, agent, data, time=None, start_time=None, metadata=None):
        return cls(agent, time, 'message', data, start_time=start_time, metadata=metadata)

    @classmethod
    def JoinEvent(cls, agent, userid=None, time=None):
        return cls(agent, time, 'join', userid)

    @classmethod
    def LeaveEvent(cls, agent, userid=None, time=None):
        return cls(agent, time, 'leave', userid)

    @classmethod
    def TypingEvent(cls, agent, data, time=None):
        return cls(agent, time, 'typing', data)

    @classmethod
    def EvalEvent(cls, agent, data, time):
        return cls(agent, time, 'eval', data)

    @staticmethod
    def gather_eval(events):
        event_dict = {e.time: e for e in events if e.action != 'eval'}
        for e in events:
            if e.action == 'eval':
                event_dict[e.time].tags = [k for k, v in e.data['labels'].iteritems() if v != 0]
            else:
                event_dict[e.time].tags = []
        events_with_eval = [v for k, v in sorted(event_dict.iteritems(), key=lambda x: x[0])]
        return events_with_eval
