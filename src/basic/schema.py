'''
A schema specifies information about a domain (types, entities, relations).
'''

import json

class Attribute(object):
    def __init__(self, name, value_type, unique):
        self.name = name
        self.value_type = value_type
        self.unique = unique
    @staticmethod
    def from_json(raw):
        return Attribute(raw['name'], raw['value_type'], raw['unique'])
    def to_json(self):
        return {'name': self.name, 'value_type': self.value_type, 'unique': self.unique}

class Schema(object):
    '''
    A schema contains information about possible entities and relations.
    '''
    def __init__(self, path):
        raw = json.load(open(path))
        # Mapping from type (e.g., hobby) to list of values (e.g., hiking)
        self.values = raw['values']
        # List of attributes (e.g., place_of_birth)
        self.attributes = [Attribute.from_json(a) for a in raw['attributes']]
