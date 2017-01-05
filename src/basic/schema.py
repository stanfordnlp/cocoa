'''
A schema specifies information about a domain (types, entities, relations).
'''

import json
import numpy as np
from itertools import izip


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
    def __init__(self, path, domain=None):
        raw = json.load(open(path))
        # Mapping from type (e.g., hobby) to list of values (e.g., hiking)
        values = raw['values']
        # List of attributes (e.g., place_of_birth)
        attributes = [Attribute.from_json(a) for a in raw['attributes']]
        self.attr_names = [attr.name for attr in attributes]

        def _get_subset(attr_names):
            subset_attributes = [attr for attr in attributes if attr.name in attr_names]
            subset_values = {}
            for attr in subset_attributes:
                k = attr.value_type
                subset_values[k] = values[k]
            return subset_attributes, subset_values

        if domain == 'Matchmaking':
            attr_names = ['Time Preference', 'Location Preference', 'Hobby']
            self.attributes, self.values = _get_subset(attr_names)
        elif domain == 'MutualFriends':
            attr_names = ['Name', 'School', 'Major', 'Company']
            self.attributes, self.values = _get_subset(attr_names)
        elif domain is None:
            # Use all attributes in the schema
            self.values = values
            self.attributes = attributes
        else:
            raise ValueError('Unknown domain.')
        self.domain = domain

        # Dirichlet alphas for scenario generation
        if domain == 'Matchmaking':
            alphas = [1.] * len(self.attributes)
        else:
            alphas = list(np.linspace(1, 0.1, len(self.attributes)))
        self.alphas = dict((attr, alpha) for (attr, alpha) in izip(self.attributes, alphas))
        for i, attr in enumerate(self.attributes):
            if attr.name == 'Name':
                self.alphas[attr] = 2
            elif attr.name == 'Hobby':
                self.alphas[attr] = 0.5
            elif attr.name == 'Time Preference':
                self.alphas[attr] = 1.0
            elif attr.name == 'Location Preference':
                self.alphas[attr] = 1.0

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
