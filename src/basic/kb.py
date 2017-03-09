from collections import defaultdict
from sample_utils import sorted_candidates
import csv
import json
import src.config as config

class BaseKB(object):
    '''
    Represents an agent's knowledge.
    '''
    def __init__(self, attributes, items):
        self.attributes = attributes
        self.items = items

    def to_dict(self):
        return self.items

    def dump(self):
        raise NotImplementedError

class KB(object):
    '''
    Factory of KBs.
    '''
    @staticmethod
    def get_kb(*args):
        if config.task == config.MutualFriends:
            return MutualFriendsKB(*args)
        elif config.task == config.Negotation:
            return NegotiationKB(*args)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

    @staticmethod
    def from_dict(attributes, raw):
        if config.task == config.MutualFriends:
            return MutualFriendsKB.from_dict(attributes, raw)
        elif config.task == config.Negotation:
            return NegotiationKB.from_dict(attributes, raw)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

class NegotiationKB(BaseKB):
    def __init__(self, attributes, items):
        super(NegotiationKB, self).__init__(attributes, items)

    @staticmethod
    def from_dict(attributes, raw):
        return NegotiationKB(attributes, raw)

    def dump(self):
        item = self.items[0]
        role = item['Role']
        price_range = (item['Bottomline'], item['Target'])
        print '----------------'
        print 'Role:', role
        print 'Price range:', price_range
        width = max([len(str(attr.name)) for attr in self.attributes])
        for attr in self.attributes:
            if attr.name not in ('Role', 'Bottomline', 'Target'):
                print '{name:<{width}s} {value}'.format(width=width, name=attr.name, value=str(item[attr.name]))


class MutualFriendsKB(BaseKB):
    def __init__(self, attributes, items):
        super(MutualFriendsKB, self).__init__(attributes, items)
        self.entity_set = set([value.lower() for item in items for value in item.values()])
        self.entity_type_set = set([attr.value_type for attr in self.attributes])

    @staticmethod
    def from_dict(attributes, raw):
        return MutualFriendsKB(attributes, raw)

    def dump(self):
        header_item = dict((attr.name, attr.name) for attr in self.attributes)
        rows = [header_item] + self.items
        widths = [max(len(str(row[attr.name])) for row in rows) for attr in self.attributes]
        print '----------------'
        for row in rows:
            print ' ', '  '.join(('%%-%ds' % widths[i]) % (row[attr.name],) for i, attr in enumerate(self.attributes))

    def get_item(self, idx):
        return self.items[idx]

