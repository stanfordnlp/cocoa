import re
import time
import spacy
from fuzzywuzzy import fuzz
import numpy as np
from nltk import ngrams, pos_tag, RegexpParser, Tree
from datasketch import MinHash, MinHashLSH
from itertools import ifilter
import nltk
from nltk.corpus import stopwords as nltk_stopwords

from cocoa.core.util import generate_uuid, write_pickle, read_pickle
from cocoa.core.entity import CanonicalEntity, Entity

def add_lexicon_arguments(parser):
    parser.add_argument('--lexicon', help='Path to pickled lexicon')

class Lexicon(object):
    # Titles that are confusing...
    stopwords = set(nltk_stopwords.words('english') +
            ['it', 'yes', 'hello', 'i', 'good morning', 'love', 'men', 'no', 'the day', 'i wish', 'lol', 'the first time', 'the dog', 'the lady'])
    nlp = spacy.load('en')

    def __init__(self, entities=None, threshold=0.8, lsh=None):
        self.entities = entities
        self.lsh = self.build_lsh(threshold) if not lsh else lsh

    def save_pickle(self, path):
        print 'Dump lexicon to {}'.format(path)
        write_pickle({'entities': self.entities, 'lsh': self.lsh}, path)

    @classmethod
    def from_pickle(cls, path):
        data = read_pickle(path)
        return cls(**data)

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
        for ss in ngrams('^^^'+s, 4):
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

    def link_entity(self, text):
        """Link tokens to entities.

        Example:
            ['i', 'work', 'at', 'apple'] =>
            ['i', 'work', 'at', ('apple', ('apple','company'))]

        """
        doc = self.nlp(unicode(text))
        entities = []
        for np in doc.noun_chunks:
            s = np.text
            candidates = self.query(s, k=1)
            if candidates:
                sorted_candidates = sorted(candidates, key=lambda x: fuzz.ratio(s.lower(), x.value.lower()), reverse=True)
                for candidate in ifilter(lambda e: e.value.lower() not in self.stopwords, sorted_candidates):
                    if fuzz.ratio(s.lower(), candidate.value.lower()) > 80:
                        entity = Entity(surface=s, canonical=candidate)
                        entities.append((entity, np.start, np.end))
                        # Take the best matched candidate
                        break

        def overlap(e, entities):
            for entity in entities:
                if not (e.start >= entity[2] or e.end < entity[1]):
                    return True
            return False

        for ent in doc.ents:
            if not overlap(ent, entities):
                if ent.label_ == 'PERSON':
                    entity = Entity.from_elements(surface=ent.text, value=ent.text, type='person')
                    entities.append((entity, ent.start, ent.end))
                elif ent.label_ == 'WORK_OF_ART':
                    entity = Entity.from_elements(surface=ent.text, value=ent.text, type='title')
                    entities.append((entity, ent.start, ent.end))

        tokens = [tok.text for tok in doc]
        if not entities:
            entity_tokens = tokens
        else:
            last = 0
            entity_tokens = []
            entities = sorted(entities, key=lambda x: x[1])
            for entity in entities:
                entity, start, end = entity
                entity_tokens.extend(tokens[last:start])
                entity_tokens.append(entity)
                last = end
            if last < len(tokens):
                entity_tokens.extend(tokens[last:])
        return entity_tokens

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--movie-data', help='Path to file of movie metadata and reviews')
    parser.add_argument('--threshold', type=float, default=0.8, help='LSH distance threshold')
    parser.add_argument('--output', help='Path to save the lexicon')
    parser.add_argument('--lexicon', help='Path to pickled lexicon')
    args = parser.parse_args()

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

    from core.tokenizer import tokenize
    sents = [
            '''i did like the mad max remake though. mad max 1 was n't that great, but mad max 2 > the remake > mad max 1 > > > > > thunderdome''',
            '''I just saw Dunkirk.''',
            '''I dont either. I also like the monty python ones''',
            '''I love that one. I also really like The Last Castle. I guess we both like prison movies!''',
            '''I haven't seen much new, but am going on Sunday to see Bad Mom's Christmas. The first one was funny. Hoping the sequel is good as well!''',
            '''No, the last good movie in theaters was Get Out by Jordan Peele.''',
            '''Nice. I love The Room, but I can't stand the one that James Franco just did.''',
            ]
    for s in sents:
        print lexicon.link_entity(s)
        break
