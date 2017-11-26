import re
import json
import time
import random
from fuzzywuzzy import fuzz
import numpy as np
from nltk import ngrams, pos_tag, RegexpParser, Tree, word_tokenize
from datasketch import MinHash, MinHashLSH

from cocoa.core.util import generate_uuid, write_pickle, read_pickle
from cocoa.core.entity import CanonicalEntity, Entity

def add_lexicon_arguments(parser):
    parser.add_argument('--movie-data', help='Path to file of movie metadata and reviews')
    parser.add_argument('--threshold', type=float, default=0.6, help='LSH distance threshold')

class Lexicon(object):
    def __init__(self, entities=None, threshold=0.6, lsh=None):
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
        new_sentence = True
        for idx, token in enumerate(tokens):
            if new_sentence:
                try:
                   a = tokens[idx+1][0]
                except(IndexError):
                   continue
                if tokens[idx+1][0].isupper() and tokens[idx+1] != "I":
                    lowered.append(token)
                else:
                    lowered.append( token.lower() )
                new_sentence = False
            elif token in ["Ive", "I've", "I", "I'm", "Im"]:
                lowered.append( token.lower() )
            else:
                lowered.append(token)
            if token in [".", "?", "!"]:
                if tokens[idx-1] not in ["Dr", "Mrs", "Ms", "Mr"]:
                    new_sentence = True
                lowered.append(token)
        return lowered

    def genres_by_keyword(self, tokens):
        keywords = []
        genres = ['comedy', 'romance', 'action', 'drama', 'sci-fi', 'comedies', 'documentary', 'documentaries', 'horror']
        for token in tokens:
            if token.lower() in genres:
                keywords.append({"GENRE": token})
        return keywords

    def titles_by_capitalization(self, tokens):
        lowered = self.lower_first(tokens)
        capitals = []
        title_flag = False
        candidate = []
        for idx, token in enumerate(lowered):
            if token in ["of", "the", "a", "an", "in", "and"] and title_flag:
                if len(lowered) > idx:
                    if lowered[idx+1][0].isupper() or lowered[idx+1] == 'the':
                        token = token.title()
            if token[0].isupper() or token[0] in ["2", "3", "4", "I"]:
                candidate.append(token)
                if not title_flag:
                    title_flag = True
            if title_flag and token[0] == "'":
                candidate.append(token)
            elif token[0].islower() and title_flag:
                title_flag = False
                c = " ".join(candidate)
                capitals.append({"TITLE": c.title()})
                candidate = []
            if (idx+1 == len(lowered)) and title_flag:
                c = " ".join(candidate)
                capitals.append({"TITLE": c.title()})
        return capitals

    def titles_by_quote(self, line):
        quoted = []
        search_obj = re.search(r"\'[^\']+\'", line)
        if search_obj is not None:
            trimmed = search_obj.group()[1:-1]
            pieces = trimmed.split()
            if len(pieces) < 5:
                quoted.append({"TITLE": trimmed.title()})
        return quoted

    def titles_by_callout(self, line):
        callout = []
        search_obj = re.search(r"(?<=a movie called )(\w*)", line)
        if search_obj is not None:
            callout.append({"TITLE": search_obj.group(0).title()})
        return callout

    def check_for_fuzz(self, parsed, debug=False):
        entity_tokens = []
        for candidate in parsed:
            key = candidate.keys()[0]
            if key == "TITLE":
                s = candidate[key]
                entities = self.query(s, k=1)
                if debug:
                    print("s: {}".format(s) )
                    print("entity: {}".format(entities) )
                if len(entities) == 0:
                    entities = self.query("The "+s, k=1)
                    if len(entities) == 0: continue
                if fuzz.ratio(s, entities[0].value.title()) > 80:
                    entity_tokens.append(Entity(surface=s, canonical=entities[0]))
                elif len(entities) > 1:
                    for idx, canon in enumerate(entities):
                        if fuzz.ratio(s, canon.value.title()) > 70:
                            entity_tokens.append(Entity(surface=s, canonical=entities[idx]))
                            break
            else:
                entity_tokens.append(candidate)
        return entity_tokens

    def link_entity(self, line, debug=False):
        tokens = word_tokenize(line)
        parsed = []
        parsed.extend( self.genres_by_keyword(tokens) )
        parsed.extend( self.titles_by_quote(line) )
        parsed.extend( self.titles_by_callout(line) )
        parsed.extend( self.titles_by_capitalization(tokens) )
        parsed = self.check_for_fuzz(parsed, debug)
        return parsed

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_lexicon_arguments(parser)
    parser.add_argument('--output', help='Path to save the lexicon')
    parser.add_argument('--lexicon', help='Path to pickled lexicon')
    parser.add_argument('--unit-test', default=False, action='store_true',
        help='if set to True, we run the full unit test')
    parser.add_argument('-n', '--num-examples', default=10, type=int,
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
        for z in random.sample(zample, args.num_examples):
            print z
            print lexicon.link_entity(z.encode('utf8'), False)
    else:
        print lexicon.link_entity('have you heard of a movie called titanic?', True)
        # print lexicon.link_entity('I just watched Planet Earth.', False)

