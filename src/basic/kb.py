from collections import defaultdict
from sample_utils import sorted_candidates

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

    def sorted_attr(self):
        counts = defaultdict(int)
        for item in self.items:
            for attr_name, attr_value in item.iteritems():
                counts[(attr_name, attr_value)] += 1
        return sorted_candidates(counts.items())

    def get_item(self, idx):
        item = self.items[idx]
        sorted_item = [(attr.name, item[attr.name]) for attr in self.schema.attributes]
        return sorted_item
