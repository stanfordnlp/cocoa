import argparse
import numpy as np
import time

from lexicon import Lexicon
from collections import defaultdict, Counter
from lexicon_utils import get_morphological_variants, get_prefixes, get_edits, get_acronyms
from schema import Schema
from src.model.vocab import is_entity


class InverseLexicon(object):
    """
    Inverse lexicon for taking a list of entity tuples and converting to a reasonable surface form
    """
    def __init__(self, schema, inverse_lexicon_data):
        self.schema = schema
        self.entities = {}  # Mapping from (canonical) entity to type (assume type is unique)
        self.word_counts = defaultdict(int)  # Counts of words that show up in entities
        self.inverse_lexicon = defaultdict(Counter)  # Mapping from entity -> list of realized variants of entity (seen in data)
        self.load_entities()
        self._process_inverse_lexicon_data(inverse_lexicon_data)

    def _process_inverse_lexicon_data(self, inverse_lexicon_data):
        """
        Process inverse lexicon data
            <entity> \t <span> \t <type>
        and generate variant frequency count
        :return:
        """
        with open(inverse_lexicon_data, "r") as f:
            for line in f:
                entity, span, type = line.split("\t")
                self.inverse_lexicon[entity][span] += 1


    def load_entities(self):
        for type_, values in self.schema.values.iteritems():
            for value in values:
                self._add_entity(type_, value.lower())


    def _add_entity(self, type, entity):
        # Keep track of number of times words in this entity show up
        if entity not in self.entities:
            for word in entity.split(' '):
                self.word_counts[word] += 1
        self.entities[entity] = type


    def lookup(self, phrase):
        return self.inverse_lexicon.get(phrase, [])


    def realize_entity(self, entity_tokens):
        return [token if not is_entity(token) else self._realize_entity(token[1]) for token in entity_tokens]

    def _realize_entity(self, entity):
        """
        Take a list of entity tuples
        :param entities: List of entity tuples to realize to some surface form
        :return:
        """
        entity, type = entity
        if type == 'item':
            return entity
        # Try checking in inverse lexicon frequency count
        if entity not in self.inverse_lexicon:
            print "Have not encountered entity %s in data..." % entity
            realized = -1
        else:
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

        if realized != -1:
            return realized
        else:
            # TODO: Modify heuristic rules when entity not found in data
            if len(entity.split()) == 1:
                return entity
            else:
                tokens = entity.split()
                realized = ""
                # Only take first two tokens if more than three
                #if len(tokens) > 3:
                #    realized = " ".join(tokens[:3])
                #else:
                for t in tokens:
                    if t.lower() == "university":
                        realized += "univ. "
                    else:
                        realized += t + " "

                realized = realized.strip()

        return realized


    def test(self):
        entities = [("amanda", "school"), ("rowan eastern college of the arts", "school"), ("bible studies", "major")]
        realized = self.realize_entity(entities)
        print realized


if __name__ == "__main__":
    # Test inverse lexicon
    parser = argparse.ArgumentParser("arguments for basic testing lexicon")
    parser.add_argument("--schema", type=str, help="path to schema to use")
    parser.add_argument("--scenarios-json", help="Json of scenario information", type=str)
    parser.add_argument("--transcripts", type=str, help="transcripts of chats")
    parser.add_argument("--inverse-lexicon-data", type=str, help="path to inverse lexicon data")

    args = parser.parse_args()

    path = args.schema
    start_build = time.time()

    schema = Schema(path)
    inv_lex = InverseLexicon(schema, args.inverse_lexicon_data)
    print "Building complete: ", time.time() - start_build
    start_test = time.time()
    inv_lex.test()

    print "Testing Complete: ", time.time() - start_test
