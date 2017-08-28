class KB(object):
    '''
    Represents an agent's knowledge.
    '''
    def __init__(self, attributes):
        self.attributes = attributes

    def dump(self):
        raise NotImplementedError
