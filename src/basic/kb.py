from collections import defaultdict
from sample_utils import sorted_candidates
import csv
import json


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
        return self.items[idx]

    def get_ordered_item(self, idx):
        item = self.items[idx]
        ordered_item = [(attr.name, item[attr.name]) for attr in self.schema.attributes]
        return ordered_item

    @classmethod
    def ordered_item_to_dict(cls, ordered_item):
        # Convert an ordered item (a list of (key, value) pairs) to a string representation
        # of the corresponding dictionary
        # e.g. [("name","Claire"),("school","Stanford University"),...]"
        d = dict((key, val) for (key,val) in ordered_item)  # {"name": "Claire", "school": "Stanford University",..}
        return json.dumps(d)  # "{\"name\": \"Claire\", \"school\": \"Stanford University\",..}"

    @classmethod
    def string_to_item(cls, str_data):
        # Convert the string representation of an item back to an ordered item (a tuple)
        # e.g. string representation: "{\"name\": \"Claire\", \"school\": \"Stanford University\",..}"
        return json.loads(str_data)
