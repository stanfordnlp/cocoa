import argparse
import time

from collections import defaultdict
from lexicon_utils import get_morphological_variants, get_prefixes, get_edits, get_acronyms
from schema import Schema


class InverseLexicon(object):
    """
    Inverse lexicon for taking a list of entity tuples and converting to a reasonable surface form
    """
    def __init__(self, schema):
        self.schema = schema
        self.entities = {}  # Mapping from (canonical) entity to type (assume type is unique)
        self.word_counts = defaultdict(int)  # Counts of words that show up in entities
        self.inverse_lexicon = defaultdict(list)  # Mapping from entity -> list of realized variants of entity (seen in data)
        self.load_entities()


    def load_entities(self):
        for type_, values in self.schema.values.iteritems():
            for value in values:
                self._add_entity(type_, value.lower())


    def _add_entity(self, type, entity):
        # Keep track of number of times words in this entity shows up
        if entity not in self.entities:
            for word in entity.split(' '):
                self.word_counts[word] += 1
        self.entities[entity] = type


    def _load_data(self):
        """
        Load transcript data to form mapping from entity to observed variants
        :return:
        """
        # TODO: What format do we expect this to be?
        raise NotImplementedError


    def lookup(self, phrase):
        return self.inverse_lexicon.get(phrase, [])


    def realize_entity(self, entities):
        """
        Take a list of entity tuples
        :param entities: List of entity tuples to realize to some surface form
        :return:
        """
        # TODO: Incorporate variants seen before in data
        realized_entities = []
        for entity, type in entities:
            if len(entity.split()) == 1:
                realized_entities.append(entity)
            else:
                tokens = entity.split()
                realized = ""
                # Only take first two tokens if more than three
                if len(tokens) > 3:
                    realized = " ".join(tokens[:2])
                else:
                    for t in tokens:
                        if t.lower() == "university":
                            realized += "univ. "
                        else:
                            realized += t + " "

                realized_entities.append(realized.strip())

        return realized_entities


    def test(self):
        entities = [("stanford university", "school"), ("rowan eastern college of the arts", "school"), ("arco", "company")]
        realized = self.realize_entity(entities)
        print realized


if __name__ == "__main__":
    # Test inverse lexicon
    parser = argparse.ArgumentParser("arguments for basic testing lexicon")
    parser.add_argument("--schema", type=str, help="path to schema to use")
    parser.add_argument("--scenarios-json", help="Json of scenario information", type=str)

    args = parser.parse_args()

    path = args.schema
    start_build = time.time()

    schema = Schema(path)
    inv_lex = InverseLexicon(schema)
    print "Building complete: ", time.time() - start_build
    start_test = time.time()
    inv_lex.test()

    print "Testing Complete: ", time.time() - start_test