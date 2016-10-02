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
    def __init__(self, path, domain=None):
        raw = json.load(open(path))
        # Mapping from type (e.g., hobby) to list of values (e.g., hiking)
        values = raw['values']
        # List of attributes (e.g., place_of_birth)
        attributes = [Attribute.from_json(a) for a in raw['attributes']]

        def _get_subset(attr_names):
            subset_attributes = [attr for attr in attributes if attr.name in attr_names]
            subset_values = {}
            for attr in subset_attributes:
                k = attr.value_type
                subset_values[k] = values[k]
            return subset_attributes, subset_values

        if domain == 'matchmaking':
            attr_names = ['Time Preference', 'Location Preference', 'Hobby']
            self.attributes, self.values = _get_subset(attr_names)
        elif domain == 'mutualfriend':
            attr_names = ['Name', 'School', 'Major', 'Company']
            self.attributes, self.values = _get_subset(attr_names)
        else:
            # Use all attributes in the schema
            self.values = values
            self.attributes = attributes
        self.domain = domain

    # NOTE: this function will be removed in the new model because a) we don't need all entities for embedding and b) all entities in the schema may not be used in some scenarios due to sampling.
    def get_entities(self):
        '''
        Return a dict {value: type} of all entities.
        '''
        return {value: type_ for type_, values in self.values.iteritems() for value in values}

    def get_attributes(self):
        '''
        Return a dict {name: value_type} of all attributes.
        '''
        return {attr.name: attr.value_type for attr in self.attributes}
