import re
import json
import time
import random
from fuzzywuzzy import fuzz
import numpy as np
from nltk import ngrams, pos_tag, RegexpParser, Tree
from datasketch import MinHash, MinHashLSH

from cocoa.core.util import generate_uuid, write_pickle, read_pickle
from cocoa.core.entity import CanonicalEntity, Entity

def add_lexicon_arguments(parser):
    parser.add_argument('--movie-data', help='Path to file of movie metadata and reviews')
    parser.add_argument('--threshold', type=float, default=0.8, help='LSH distance threshold')

class Lexicon(object):
    def __init__(self, entities=None, threshold=0.8, lsh=None):
        self.entities = entities
        self.lsh = self.build_lsh(threshold) if not lsh else lsh
        self.chunker = self.build_np_chunker()

    def save_pickle(self, path):
        print 'Dump lexicon to {}'.format(path)
        write_pickle({'entities': self.entities, 'lsh': self.lsh}, path)

    @classmethod
    def from_pickle(cls, path):
        data = read_pickle(path)
        return cls(**data)

    @classmethod
    def build_np_chunker(cls):
        grammar = r"""
          NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}
        """
        return RegexpParser(grammar)

    @classmethod
    def from_csv(cls, path, threshold):
        entities = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                entities.append(CanonicalEntity(value=row['title'], type='title'))
        return cls(entities=entities, threshold=threshold)

    @classmethod
    def from_json(cls, path, threshold):
        entities = []
        reader = json.load( open(path, 'r') )
        for i, row in enumerate(reader):
            entities.append(CanonicalEntity(value=row['title'], type='title'))
        return cls(entities=entities, threshold=threshold)

    @classmethod
    def minhash(cls, s):
        m = MinHash(num_perm=128)
        s = re.sub(r'[^a-z0-9]', '', s.lower())
        for ss in ngrams(s, 4):
            m.update(''.join(ss))
        return m

    def build_lsh(self, threshold=0.5):
        start = time.time()
        print 'Buidling LSH...'
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        with lsh.insertion_session() as session:
            for i, entity in enumerate(self.entities):
                session.insert(i, self.minhash(entity.value))
        print '[{} s]'.format(time.time() - start)
        return lsh

    def query(self, s, k=1):
        """Return the top-`k` matched entities to string `s`.
        """
        q = self.minhash(s)
        entity_ids = self.lsh.query(q)
        return [self.entities[id_] for id_ in entity_ids]

    def chunk(self, tokens):
        tagged_tokens = pos_tag(tokens)
        tree = self.chunker.parse(tagged_tokens)
        chunks = []
        for node in tree:
            if isinstance(node, Tree) and node.label() == 'NP':
                tags = [x[1] for x in node.leaves()]
                type_ = 'entity' if 'NNP' in tags else 'NP'
                yield ([x[0] for x in node.leaves()], type_)
            else:
                yield node[0]

    def lower_first(self, tokens):
        lowered = []
        new_sentence = False
        for idx, token in enumerate(tokens):
            if idx == 0:
                lowered.append( token.lower() )
            elif new_sentence:
                lowered.append( token.lower() )
            else:
                lowered.append(token)
            if token in [".", "?", "!"]:
                new_sentence = True
        return lowered

    def link_entity(self, tokens, dry_run=False):
        """Link tokens to entities.

        Example:
            ['i', 'work', 'at', 'apple'] =>
            ['i', 'work', 'at', ('apple', ('apple','company'))]

        """
        # Capitalize 'i' so that it's tagged correctly
        tokens = self.lower_first(tokens)
        entity_tokens = []
        for chunk in self.chunk(tokens):
            if isinstance(chunk, tuple):
                chunk, type_ = chunk
                s = ' '.join(chunk)
                entity = self.query(s, k=1)
                # entity[0] = highest ranking predicted movie title
                if entity and fuzz.ratio(s.lower(), entity[0].value.lower()) > 70:
                    if dry_run:
                        entity_tokens.append("<TITLE: {}>".format(s))
                    else:
                        entity_tokens.append(Entity(surface=s, canonical=entity[0]))
                # elif type_ == 'entity':
                #     entity_tokens.append(Entity.from_elements(surface=s, value=s, type='unknown'))
                else:
                    entity_tokens.extend(chunk)
            else:
                entity_tokens.append(chunk)
        raised_tokens = ['I' if x == 'i' else x for x in entity_tokens]
        return raised_tokens

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_lexicon_arguments(parser)
    parser.add_argument('--output', help='Path to save the lexicon')
    parser.add_argument('--lexicon', help='Path to pickled lexicon')
    parser.add_argument('--unit-test', default=False, action='store_true',
        help='if set to True, we run the full unit test')
    parser.add_argument('-n', '--num_examples', default=10, type=int,
        help='number of random examples to run for unit test')
    args = parser.parse_args()
    # python core/lexicon.py --lexicon data/lexicon.pkl --unit-test

    if args.lexicon:
        lexicon = Lexicon.from_pickle(args.lexicon)
    else:
        file_type = args.movie_data.split(".")[-1]
        if file_type == "csv":
            import csv
            lexicon = Lexicon.from_csv(args.movie_data, args.threshold)
        elif file_type == "json":
            import json
            lexicon = Lexicon.from_json(args.movie_data, args.threshold)
        lexicon.save_pickle(args.output)

    if args.unit_test == True:
        zample = json.load(open("data/full_zample.json", "r"))
        nlp = spacy.load('en_core_web_sm')
        for z in random.sample(zample, 5):
            print z
            doc = nlp(z)
            print list(doc.ents)
            # print lexicon.link_entity(z.split(), True)
    else:
        tokens = 'I just watched the Planet Earth'.split()
        print lexicon.link_entity(tokens)

