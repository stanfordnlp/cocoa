from collections import namedtuple

class CanonicalEntity(namedtuple('CanonicalEntity', ['value', 'type'])):
    __slots__ = ()
    def __str__(self):
        return '[%s]' % str(self.value)

class Entity(namedtuple('Entity', ['surface', 'canonical'])):
    __slots__ = ()
    def __str__(self):
        return '[%s]' % str(self.surface)

def is_entity(x):
    return isinstance(x, Entity) or isinstance(x, CanonicalEntity)
