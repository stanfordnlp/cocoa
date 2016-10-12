class Event(object):
    '''
    An atomic event of a dialogue, which could be someone talking or making a selection.
    '''
    def __init__(self, agent, time, action, data):
        self.agent = agent
        self.time = time
        self.action = action
        self.data = data
        # tokenize, entity linking
        self.processed_data = None

    @staticmethod
    def from_dict(raw):
        return Event(raw['agent'], raw['time'], raw['action'], raw['data'])
    def to_dict(self):
        return {'agent': self.agent, 'time': self.time, 'action': self.action, 'data': self.data}

    @staticmethod
    def MessageEvent(agent, data, time=None):
        return Event(agent, time, 'message', data)

    @staticmethod
    def SelectionEvent(agent, data, time=None):
        return Event(agent, time, 'select', data)

    @staticmethod
    def JoinEvent(agent, userid=None, time=None):
        return Event(agent, time, 'join', userid)

    @staticmethod
    def LeaveEvent(agent, userid=None, time=None):
        return Event(agent, time, 'leave', userid)
