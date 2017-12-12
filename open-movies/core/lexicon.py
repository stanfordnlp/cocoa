import re
import json
import time
import random
import numpy as np
import sys

from fuzzywuzzy import fuzz
from nltk import ngrams, pos_tag, RegexpParser, Tree, word_tokenize
from datasketch import MinHash, MinHashLSH
from cocoa.core.util import generate_uuid, write_pickle, read_pickle, read_json

from cocoa.core.entity import CanonicalEntity, MovieEntity

def add_lexicon_arguments(parser):
    parser.add_argument('--movie-data', help='Path to file of movie metadata and reviews')
    parser.add_argument('--threshold', type=float, default=0.6, help='LSH distance threshold')

class Lexicon(object):
    def __init__(self, entities=None, threshold=0.6, lsh=None):
        self.entities = entities
        self.lsh = self.build_lsh(threshold) if not lsh else lsh
        self.chunker = self.build_np_chunker()

        people = read_json('data/people.json')
        self.actors = people['actors']
        self.actresses = people['actresses']
        self.genres = ['comedy', 'comedies', 'romance', 'action',
            'drama', 'sci-fi', 'documentary', 'documentaries',
            'horror', 'animation', 'scifi', 'fantasy', 'romantic']

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
        print 'Building LSH...'
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
        entity_tokens = []
        for idx, raw_token in enumerate(tokens):
            token = raw_token.lower()
            if token in self.genres:
                ent = MovieEntity.from_elements(surface=raw_token, value=token, type='genre', start=idx)
                entity_tokens.append(ent)
        return entity_tokens

    def people_by_database(self, tokens):
        entity_tokens = []
        idx = 0
        for first, second in zip(tokens[:-1], tokens[1:]):
            raw_name = " ".join([first, second])
            name = raw_name.title()
            if name in self.actors:
                ent = MovieEntity.from_elements(surface=raw_name, value=name, type='actor', start=idx)
                entity_tokens.append(ent)
            if name in self.actresses:
                ent = MovieEntity.from_elements(surface=raw_name, value=name, type='actress', start=idx)
                entity_tokens.append(ent)
            idx += 1
        if len(tokens) > 3:
            idx = 0
            for first, second, third in zip(tokens[:-2], tokens[1:], tokens[2:]):
                raw_name = " ".join([first, second, third])
                name = raw_name.title()
                if name in self.actors:
                    ent = MovieEntity.from_elements(surface=raw_name, value=name, type='actor', start=idx)
                    entity_tokens.append(ent)
                if name in self.actresses:
                    ent = MovieEntity.from_elements(surface=raw_name, value=name, type='actress', start=idx)
                    entity_tokens.append(ent)
                idx += 1
        return entity_tokens

    def people_by_phrase(self, tokens):
        entity_tokens = []
        actor_phrases = ["his acting", "best actor", "favorite actor"]
        actress_phrases = ["her acting", "best actress", "favorite actress"]
        idx = 0
        for first, second in zip(tokens[:-1], tokens[1:]):
            phrase = " ".join([first, second])
            if phrase in actor_phrases:
                ent = MovieEntity.from_elements(surface=phrase, type='actor', start=idx)
                entity_tokens.append(ent)
            if phrase in actress_phrases:
                ent = MovieEntity.from_elements(surface=phrase, type='actress', start=idx)
                entity_tokens.append(ent)
            idx += 1
        return entity_tokens

    def titles_by_capitalization(self, tokens):
        entity_tokens = []
        lowered = self.lower_first(tokens)
        title_flag = False
        candidate = []
        start_index = None
        for idx, token in enumerate(lowered):
            if token in ["of", "the", "a", "an", "in", "and"] and title_flag:
                if len(lowered) > idx:
                    if lowered[idx+1][0].isupper() or lowered[idx+1] == 'the':
                        token = token.title()
            if token[0].isupper() or token[0] in ["2", "3", "4", "I"]:
                candidate.append(token)
                if not title_flag:
                    title_flag = True
                    start_index = idx
            if title_flag and token[0] == "'":
                candidate.append(token)
            elif token[0].islower() and title_flag:
                title_flag = False
                c = " ".join(candidate)
                if self.check_for_match(c.title()):
                    ent = MovieEntity.from_elements(surface=c, value=c.title(),
                            type='title', start=start_index)
                    entity_tokens.append(ent)
                candidate = []
            if (idx+1 == len(lowered)) and title_flag:
                c = " ".join(candidate)
                if self.check_for_match(c.title()):
                    ent = MovieEntity.from_elements(surface=c, value=c.title(),
                            type='title', start=start_index)
                    entity_tokens.append(ent)
        return entity_tokens

    def titles_by_quote(self, line):
        entity_tokens = []
        search_obj = re.search(r"\'[^\']+\'", line)
        if search_obj:
            candidate = search_obj.group()[1:-1]
            pieces = candidate.split()
            match = self.check_for_match(candidate.title())
            if (len(pieces) < 5) and match:
                cleaned_tokens = line.replace("'", "").split()
                index = self.find_start_idx(candidate, cleaned_tokens)
                ent = MovieEntity.from_elements(surface=candidate,
                            value=match, type='title', start=index)
                entity_tokens.append(ent)
        return entity_tokens

    def titles_by_callout(self, line, tokens):
        entity_tokens = []
        search_obj = re.search(r"(?<=a movie called )(\w*)", line)
        if search_obj:
            candidate = search_obj.group(0)
            match = self.check_for_match(candidate.title())
            if match:
                index = self.find_start_idx(candidate, tokens)
                ent = MovieEntity.from_elements(surface=candidate,
                            value=match, type='title', start=index)
                entity_tokens.append(ent)
        return entity_tokens

    def find_start_idx(self, candidate, tokens):
        pieces = candidate.split()
        title_length = len(pieces)
        if title_length == 1:
            return tokens.index(candidate)

        first_word = pieces[0]
        matches = []
        for idx, token in enumerate(tokens):
            last_idx = idx + title_length
            if (first_word == token) and (last_idx <= len(tokens)):
                match = " ".join([tokens[x] for x in range(idx, last_idx)])
                matches.append((match, idx))
        if len(matches) == 1:
            return matches[0][1]

        for match in matches:
            if candidate == match[0]:
                return match[1]
        print "Start index not found, should never occur."
        return 0

    def check_for_match(self, candidate):
        match = False
        entities = self.query(candidate, k=1)
        # print("candidate: {}".format(s) )
        # print("entity: {}".format(entities) )
        if len(entities) == 0:
            entities = self.query("The "+candidate, k=1)
            if len(entities) == 0: return False
        if fuzz.ratio(candidate, entities[0].value.title()) > 80:
            match = entities[0].value.title()
        elif len(entities) > 1:
            for idx, canon in enumerate(entities):
                if fuzz.ratio(candidate, canon.value.title()) > 70:
                    match = canon.value.title()
                    break
        return match

    def replace_surface_form(self, entities, tokens):
      for entity in entities:
        surface_tokens = word_tokenize(entity[0])
        steps = len(surface_tokens)
        start_idx = entity[2]
        tokens[start_idx] = entity

        post_idx = start_idx + 1
        end_idx = start_idx + steps
        for step in range(post_idx, end_idx):
          tokens[step] = "<REMOVE>"

      tokens = [x for x in tokens if x != "<REMOVE>"]
      return tokens

    def link_entity(self, line):
        tokens = word_tokenize(line)
        entities = []

        # Detect genres
        entities.extend( self.genres_by_keyword(tokens) )
        # Detect actors and actresses
        entities.extend( self.people_by_database(tokens) )
        entities.extend( self.people_by_phrase(tokens) )
        # Detect movie titles
        entities.extend( self.titles_by_quote(line) )
        entities.extend( self.titles_by_callout(line, tokens) )
        entities.extend( self.titles_by_capitalization(tokens) )

        return self.replace_surface_form(entities, tokens)

    def unit_test(self):
        examples = ["Have you see the new Zootopia movie yet ?",
            "Have you heard of a movie called titanic?",
            "I just watched 'beauty and the beast' with emma watson .",
            "Well, I prefer comedies like Caddyshack . What about you ?",
            "My favorite actor is Tom Hanks, he was great in Toy Story as the sheriff ."
            ]
        for idx, example in enumerate(examples):
            print("Example {} ---------- ".format(idx+1) )
            print example
            print self.link_entity(example)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_lexicon_arguments(parser)
    parser.add_argument('--output', help='Path to save the lexicon')
    parser.add_argument('--lexicon', help='Path to pickled lexicon')
    args = parser.parse_args()
    # python core/lexicon.py --lexicon data/lexicon.pkl

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
    lexicon.unit_test()