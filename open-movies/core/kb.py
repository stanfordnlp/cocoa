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

    @classmethod
    def from_ints(cls, attributes, names, ints):
        """Build KB from integers.

        Args:
            names (list[str])
            ints (list[int]): [count1, value1, count2, value2, ...]

        """
        items = []
        assert 1. * len(ints) / len(names) == 2
        for i, name in enumerate(names):
            item = {'Name': name, 'Count': ints[i*2], 'Value': ints[i*2+1]}
            items.append(item)
        return cls(attributes, items)

    def dump(self):
        print 'Topic: {}'.format(self.topic)