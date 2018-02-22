from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, personas):
        super(KB, self).__init__(attributes)
        self.personas = personas

    def to_dict(self):
        return self.personas

    @classmethod
    def from_dict(cls, attributes, raw):
        return cls(attributes, raw)

    def dump(self):
        print "--- My Persona Traits ---"
        for p in self.personas:
            print p