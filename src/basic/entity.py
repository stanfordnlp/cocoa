from collections import namedtuple

CanonicalEntity = namedtuple('CanonicalEntity', ['value', 'type'])
Entity = namedtuple('Entity', ['surface', 'canonical'])

def is_entity(x):
    return isinstance(x, Entity) or isinstance(x, CanonicalEntity)
