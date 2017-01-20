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

    def get_item_id(self, item):
        for i, it in enumerate(self.items):
            if item == it:
                return i
        return None

    def get_ordered_item(self, item):
        ordered_item = [(attr.name, item[attr.name]) for attr in self.attributes]
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
