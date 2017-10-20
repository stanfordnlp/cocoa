from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, topic):
        super(KB, self).__init__(attributes)
        self.topic = topic

    def to_dict(self):
        return self.topic

    @classmethod
    def from_dict(cls, attributes, raw):
        return cls(attributes, raw)

    def dump(self):
        print 'Topic: {}'.format(self.topic)
