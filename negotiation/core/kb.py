from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, facts):
        super(KB, self).__init__(attributes)
        self.facts = facts

    def to_dict(self):
        return self.facts

    @staticmethod
    def from_dict(attributes, raw):
        return KB(attributes, raw)

    def dump(self):
        personal_info = self.facts['personal']
        role = personal_info['Role']
        price_range = (personal_info['Bottomline'], personal_info['Target'])
        print '----------------'
        print 'Listing price:', self.facts['item']['Price']
        print 'Role:', role
        print 'Price range:', price_range
        width = max([len(str(attr.name)) for attr in self.attributes])
        for attr in self.attributes:
            if attr.name not in ('Role', 'Bottomline', 'Target'):
                if attr.name == 'Description':
                    value = '\n' + '\n'.join(self.facts['item'][attr.name]).encode('utf8')
                elif attr.name == 'Price':
                    value = self.facts['item'][attr.name]
                elif attr.name == 'Images':
                    value = ' '.join(self.facts['item'][attr.name])
                else:
                    value = self.facts['item'][attr.name].encode('utf8')
                print '{name:<{width}s} {value}'.format(width=width, name=attr.name, value=value)
