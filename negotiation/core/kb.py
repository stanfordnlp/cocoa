from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, facts):
        super(KB, self).__init__(attributes)
        self.facts = facts
        self._title = self.facts['item']['Title']
        self._category = self.facts['item']['Category']
        self._role = self.facts['personal']['Role']
        self._listing_price = self.facts['item']['Price']
        self._target = self.facts['personal']['Target']

    @property
    def listing_price(self):
        return self._listing_price

    @property
    def target(self):
        return self._target

    @property
    def category(self):
        return self._category

    @property
    def title(self):
        return self._title

    @property
    def role(self):
        return self._role

    def to_dict(self):
        return self.facts

    @classmethod
    def from_dict(cls, attributes, raw):
        return cls(attributes, raw)

    def dump(self):
        personal_info = self.facts['personal']
        role = self._role
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
