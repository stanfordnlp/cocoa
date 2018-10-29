from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, facts):
        super(KB, self).__init__(attributes)
        self.facts = facts

    @property
    def listing_price(self):
        return self.facts['item']['Price']

    @property
    def target(self):
        return self.facts['personal']['Target']

    @property
    def category(self):
        return self.facts['item']['Category']

    @property
    def title(self):
        return self.facts['item']['Title']

    @property
    def role(self):
        return self.facts['personal']['Role']

    def to_dict(self):
        return self.facts

    @classmethod
    def from_dict(cls, attributes, raw):
        return cls(attributes, raw)

    def dump(self):
        # NOTE: We no longer have a bottomline price
        price_range = (None, self.target)
        print('----------------')
        print('Role: {}'.format(self.role))
        print('Price range: {}'.format(str(price_range)))
        if self.role == 'seller':
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
                    print('{empty:4}{name:<{width}s} {value}'.format(empty='', width=width, name=attr.name, value=value))
