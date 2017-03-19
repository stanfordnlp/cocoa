from collections import defaultdict
from sample_utils import sorted_candidates
import csv
import json
import src.config as config

class BaseKB(object):
    '''
    Represents an agent's knowledge.
    '''
    def __init__(self, attributes):
        self.attributes = attributes

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
        elif config.task == config.Negotiation:
            return NegotiationKB(*args)
        elif config.task == config.Party:
            return PartyKB(*args)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

    @staticmethod
    def from_dict(attributes, raw):
        if config.task == config.MutualFriends:
            return MutualFriendsKB.from_dict(attributes, raw)
        elif config.task == config.Negotiation:
            return NegotiationKB.from_dict(attributes, raw)
        elif config.task == config.Party:
            return PartyKB.from_dict(attributes, raw)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

class NegotiationKB(BaseKB):
    def __init__(self, attributes, facts):
        super(NegotiationKB, self).__init__(attributes)
        self.facts = facts

    def to_dict(self):
        return self.facts

    @staticmethod
    def from_dict(attributes, raw):
        return NegotiationKB(attributes, raw)

    def dump(self):
        personal_info = self.facts['personal']
        role = personal_info['Role']
        price_range = (personal_info['Bottomline'], personal_info['Target'])
        print '----------------'
        print 'Role:', role
        print 'Price range:', price_range
        width = max([len(str(attr.name)) for attr in self.attributes])
        for attr in self.attributes:
            if attr.name not in ('Role', 'Bottomline', 'Target'):
                print '{name:<{width}s} {value}'.format(width=width, name=attr.name, value=str(self.facts['item'][attr.name]))

class PartyKB(BaseKB):
    def __init__(self, attributes, disc_params):
        """
        disc_params: Includes discussion constraints which might be expanded to include topics, events etc.
        """
        super(PartyKB, self).__init__(attributes)
        self.disc_params = disc_params

    def to_dict(self):
        return self.disc_params

    @staticmethod
    def from_dict(attributes, raw):
        return PartyKB(attributes, raw)

    def dump(self):
        print '----------------'
        width = max([len(str(attr.name)) for attr in self.attributes])
        for attr in self.attributes:
            print '{name:<{width}s} {value}'.format(width=width, name=attr.name, value=str(self.disc_params[attr.name]))

class MutualFriendsKB(BaseKB):
    def __init__(self, attributes, items):
        super(MutualFriendsKB, self).__init__(attributes)
        self.items = items
        self.entity_set = set([value.lower() for item in items for value in item.values()])
        self.entity_type_set = set([attr.value_type for attr in self.attributes])

    def to_dict(self):
        return self.items

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

