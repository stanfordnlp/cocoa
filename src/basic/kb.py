from collections import defaultdict
from sample_utils import sorted_candidates
import csv
import json


class KB(object):
    '''
    Represents an agent's knowledge.
    '''
    def __init__(self, attributes, items):
        self.attributes = attributes
        self.items = items
        self.entity_set = set([value.lower() for item in items for value in item.values()])
        self.entity_type_set = set([attr.value_type for attr in self.attributes])

    @staticmethod
    def from_dict(attributes, raw):
        return KB(attributes, raw)

    def to_dict(self):
        return self.items

    def dump(self):
        header_item = dict((attr.name, attr.name) for attr in self.attributes)
        rows = [header_item] + self.items
        widths = [max(len(str(row[attr.name])) for row in rows) for attr in self.attributes]
        print '----------------'
        for row in rows:
            print ' ', '  '.join(('%%-%ds' % widths[i]) % (row[attr.name],) for i, attr in enumerate(self.attributes))

    def get_item(self, idx):
        return self.items[idx]
