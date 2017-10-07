'''
A schema specifies information about a domain (types, entities, relations).
'''

import json
from itertools import izip


class Attribute(object):
    def __init__(self, name, value_type, unique=False, multivalued=False, entity=True):
        self.name = name
        self.value_type = value_type
        self.unique = unique
        self.multivalued = multivalued
        # Whether the value of this attribute is an entity
        self.entity = entity

    @staticmethod
    def from_json(raw):
        return Attribute(raw['name'], raw['value_type'], raw.get('unique', False), raw.get('multivalued', False), raw.get('entity', True))

    def to_json(self):
        return {'name': self.name, 'value_type': self.value_type, 'unique': self.unique, 'multivalued': self.multivalued, 'entity': self.entity}


class Schema(object):
    '''
    A schema contains information about possible entities and relations.
    '''
    def __init__(self, path, domain=None):
        raw = json.load(open(path))
        # Mapping from type (e.g., hobby) to list of values (e.g., hiking)
        values = raw['values']
        # List of attributes (e.g., place_of_birth)
        attributes = [Attribute.from_json(a) for a in raw['attributes']]
        self.attr_names = [attr.name for attr in attributes]

        self.values = values
        self.attributes = attributes
        self.domain = domain

    def get_attributes(self):
        '''
        Return a dict {name: value_type} of all attributes.
        '''
        return {attr.name: attr.value_type for attr in self.attributes}

    def get_ordered_attribute_subset(self, attribute_subset):
        """
        Order a subset of this schema's attributes using the original order of attributes in the schema.
        attribute_subset: A list containing the names of the attributes present in the subset
        :return The same list, preserving the original order of attributes in this schema
        """

        subset_ordered = sorted([(attr, self.attributes.index(attr)) for attr in attribute_subset], key=lambda x: x[1])

        return [x[0] for x in subset_ordered]

    def get_ordered_item(self, item):
        '''
        Order attrs in item according to get_ordered_attribute_subset and return a list.
        '''
        ordered_item = []
        for name in self.attr_names:
            try:
                ordered_item.append((name, item[name]))
            except KeyError:
                continue
        return ordered_item
