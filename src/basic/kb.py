from collections import defaultdict
from sample_utils import sorted_candidates
import csv


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

    def get_item(self, idx):
        return self.items[idx]

    def get_ordered_item(self, idx):
        item = self.items[idx]
        ordered_item = [(attr.name, item[attr.name]) for attr in self.schema.attributes]
        return ordered_item

    @classmethod
    def ordered_item_to_string(cls, ordered_item):
        # Convert an ordered item (a tuple) to a string
        str_items = ["'\"%s\":\"%s\"'" % (i[0], i[1]) for i in ordered_item]  # [ '"name":"Claire"', '"school":"Stanford University"',...]
        return ",".join(str_items)  # '"name":"Claire","school":"Stanford University",..."

    @classmethod
    def string_to_ordered_item(cls, str_data):
        # Convert the string representation of an item back to an ordered item (a tuple)
        str_data = [str_data]
        reader = csv.reader(str_data, quotechar="'", delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        str_items = next(reader)
        formatted_items = [x.split(":") for x in str_items]
        data = [(x[0].strip('"'), x[1].strip('"')) for x in formatted_items]
        return data
