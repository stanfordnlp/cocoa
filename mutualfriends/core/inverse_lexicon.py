import argparse
import time
import re
import numpy as np
from collections import defaultdict, Counter
from itertools import ifilter

from cocoa.core.entity import is_entity, CanonicalEntity, Entity
from cocoa.core.schema import Schema

class DefaultInverseLexicon(object):
    """Dumb inverse lexicon that returns the canonical value.
    """
    def realize_entity(self, entity_tokens):
        return [token if not is_entity(token) or token.type == 'item'
                else self._realize_entity(token).surface
                for token in entity_tokens]

    def _realize_entity(self, entity):
        if isinstance(entity, Entity):
            entity = entity.canoncial
        elif isinstance(entity, CanonicalEntity):
            pass
        else:
            raise TypeError('Unknown entity')

        s = re.sub(r',|-|&', ' ', entity.value)
        tokens = [tok for tok in ifilter(lambda x: x.lower() not in ('the', 'of', 'and'), s.split())]
        if entity.type == 'school':
            tokens = [tok for tok in ifilter(lambda x: x.lower() not in ('university', 'college', 'state', 'at'), tokens)]
        elif entity.type == 'company':
            tokens = [tok for tok in ifilter(lambda x: x.lower() not in ('at', 'company', 'corporation', 'group'), tokens)]
        surface = ' '.join(tokens[:2])
        return Entity(surface, entity)


class InverseLexicon(DefaultInverseLexicon):
    """Inverse lexicon that chooses a surface form based on those seen in the training data.
    """
    def __init__(self, inverse_lexicon):
        # Mapping from entity -> list of realized variants of entity (seen in data)
        self.inverse_lexicon = inverse_lexicon

    @classmethod
    def from_file(cls, inverse_lexicon_path):
        """Read linked entities from file.

        Process inverse lexicon data
            <entity> \t <span> \t <type>
        and generate variant frequency count

        """
        inverse_lexicon = defaultdict(Counter)
        with open(inverse_lexicon_path, "r") as f:
            for line in f:
                value, surface, type_ = line.strip().split("\t")
                entity = CanonicalEntity(value, type_)
                inverse_lexicon[entity][surface] += 1
        return cls(inverse_lexicon)

    def _realize_entity(self, entity):
        """Return an observed surface form of entity.

        Args:
            entity (Entity or CanonicalEntity)

        """
        entity = super(InverseLexicon, self)._realize_entity(entity)

        if entity.canonical not in self.inverse_lexicon:
            return entity
        else:
            entity = entity.canonical
            items = self.inverse_lexicon[entity].items()
            variants = [item[0] for item in items]
            counts = np.array([item[1] for item in items], dtype=np.float32)
            # Make it peaky
            peaky_counts = counts ** 2
            normal_counts = peaky_counts / np.sum(peaky_counts)
            try:
                idx = np.random.choice(np.arange(len(counts)), 1, p=normal_counts)[0]
            except ValueError:
                idx = np.argmax(counts)
            realized = variants[idx]
            return Entity(realized, entity)

    def test(self):
        entities = [CanonicalEntity("amanda", "name"), CanonicalEntity("rowan eastern college of the arts", "school"), CanonicalEntity("bible studies", "major")]
        realized = self.realize_entity(entities)
        print realized


if __name__ == "__main__":
    # Test inverse lexicon
    parser = argparse.ArgumentParser("arguments for basic testing lexicon")
    parser.add_argument("--inverse-lexicon-data", type=str, help="path to inverse lexicon data")
    args = parser.parse_args()

    start_build = time.time()
    inv_lex = InverseLexicon.from_file(args.inverse_lexicon_data)
    print "Building complete: ", time.time() - start_build

    start_test = time.time()
    inv_lex.test()
    print "Testing Complete: ", time.time() - start_test
