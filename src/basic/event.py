class Event(object):
    """
    An atomic event of a dialogue, which could be someone talking or making a selection.
    """
    def __init__(self, agent, time, action, data, start_time=None, metadata=None):
        '''
        Params:
        agent: The index of the agent triggering the event
        time: Time at which event occurred
        action: The action this event corresponds to ('select', 'message', ..)
        data: Any data that is part of the event
        start_time: The time at which the event action was started (e.g. the time at which an agent starting typing a
        message to send)
        :param metadata: Any additional metadata that needs to be stored in the event.
        :return:
        '''
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

    @staticmethod
    def MessageEvent(agent, data, time=None, start_time=None, metadata=None):
        return Event(agent, time, 'message', data, start_time=start_time, metadata=metadata)

    @staticmethod
    def SelectionEvent(agent, data, time=None, metadata=None):
        return Event(agent, time, 'select', data, metadata)

    @staticmethod
    def JoinEvent(agent, userid=None, time=None, metadata=None):
        return Event(agent, time, 'join', userid, metadata)

    @staticmethod
    def LeaveEvent(agent, userid=None, time=None, metadata=None):
        return Event(agent, time, 'leave', userid, metadata)
