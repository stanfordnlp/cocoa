from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, items):
        super(KB, self).__init__(attributes)
        self.items = items
        self.entity_set = set([value.lower() for item in items for value in item.values()])
        self.entity_type_set = set([attr.value_type for attr in self.attributes])

    def to_dict(self):
        return self.items

    @staticmethod
    def from_dict(attributes, raw):
        return KB(attributes, raw)

    def dump(self):
        header_item = dict((attr.name, attr.name) for attr in self.attributes)
        rows = [header_item] + self.items
        widths = [max(len(str(row[attr.name])) for row in rows) for attr in self.attributes]
        print '----------------'
        for i, row in enumerate(rows):
            id_ = '{:3s}'.format('') if i == 0 else '{:<3d}'.format(i-1)
            print id_, ' ', '  '.join(('%%-%ds' % widths[i]) % (row[attr.name],) for i, attr in enumerate(self.attributes))

    def get_item(self, idx):
        return self.items[idx]
