class KB(object):
    '''
    Represents an agent's knowledge.
    '''
    def __init__(self, schema, items):
        self.schema = schema
        self.items = items

    @staticmethod
    def from_dict(schema, raw):
        return KB(schema, raw)
    def to_dict(self):
        return self.items

    def dump(self):
        header_item = dict((attr.name, attr.name) for attr in self.schema.attributes)
        rows = [header_item] + self.items
        widths = [max(len(str(row[attr.name])) for row in rows) for attr in self.schema.attributes]
        print '----------------'
        for row in rows:
            print ' ', '  '.join(('%%-%ds' % widths[i]) % (row[attr.name],) for i, attr in enumerate(self.schema.attributes))
