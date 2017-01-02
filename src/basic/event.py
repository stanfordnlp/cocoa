class Event(object):
    """
    An atomic event of a dialogue, which could be someone talking or making a selection.
    """
    def __init__(self, agent, time, action, data, metadata=None):
        '''
        Creates a new Event object. The static functions MessageEvent, SelectionEvent, etc. should be used for standard
        event types.
        :param agent: The agent that triggered the event
        :param time: Time at which the event was triggered. This *must* be a UNIX timestamp.
        :param action: The type of action that the event represents (e.g. 'select', 'message')
        :param data: The data contained in the event (e.g. the selected item, the message that was sent)
        :param metadata: Any additional metadata that needs to be stored in the event.
        :return:
        '''
        self.agent = agent
        self.time = time
        self.action = action
        self.data = data
        self.metadata = metadata

    @staticmethod
    def from_dict(raw):
        return Event(raw['agent'], raw['time'], raw['action'], raw['data'], raw.get('metadata'))

    def to_dict(self):
        return {'agent': self.agent, 'time': self.time, 'action': self.action, 'data': self.data,
                'metadata': self.metadata}

    @staticmethod
    def MessageEvent(agent, data, time=None, metadata=None):
        return Event(agent, time, 'message', data, metadata)

    @staticmethod
    def SelectionEvent(agent, data, time=None, metadata=None):
        return Event(agent, time, 'select', data, metadata)

    @staticmethod
    def JoinEvent(agent, userid=None, time=None, metadata=None):
        return Event(agent, time, 'join', userid, metadata)

    @staticmethod
    def LeaveEvent(agent, userid=None, time=None, metadata=None):
        return Event(agent, time, 'leave', userid, metadata)
