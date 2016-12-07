from collections import defaultdict
import datetime
### Helper functions

# TODO: hack for dubious synonyms (went, friends) before we have a better lexicon
# TODO: common words in this corpus
#with open('data/common_words.txt') as fin:
#    common_words = set([w.strip() for w in fin.read().split()[:2000] if len(w.strip()) > 1])

def get_prefixes(entity, min_length=3, max_length=5, common_words=[]):
    # computer science => ['comp sci', ...]
    words = entity.split()
    candidates = ['']
    for word in words:
        new_candidates = []
        for c in candidates:
            if len(word) < max_length:  # Keep word
                new_candidates.append(c + ' ' + word)
            else:  # Shorten
                for i in range(min_length, max_length):
                    new_candidates.append(c + ' ' + word[:i])
        candidates = new_candidates
    # TODO: hack for false positives
    return [c[1:] for c in candidates if c[1:] != entity and c[1:] not in common_words]

def get_acronyms(entity, common_words):
    words = entity.split()
    if len(words) < 2:
        return []
    acronyms = [''.join([w[0] for w in words])]
    if 'of' in words:
        # handle 'u of p'
        acronym = ''
        for w in words:
            acronym += w[0] if w != 'of' else ' '+w+' '
        acronyms.append(acronym)
        # handle 'upenn'
        acronym = ''
        for w in words[:-1]:
            acronym += w[0] if w != 'of' else ''
        acronym += words[-1][:4]
        acronyms.append(acronym)

    # TODO: hack for false positives
    acronyms = [w for w in acronyms if w not in common_words]
    return acronyms

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', ' ']
def get_edits(entity):
    if len(entity) < 3:
        return []
    edits = []
    for i in range(len(entity) + 1):
        prefix = entity[:i]
        # Insert
        suffix = entity[i:]
        for c in alphabet:
            new_word = prefix + c + suffix
            edits.append(new_word)

        if i == len(entity):
            continue

        # Delete
        suffix = entity[i+1:]
        new_word = prefix + suffix
        edits.append(new_word)

        # Substitute
        suffix = entity[i+1:]
        for c in alphabet:
            if c != entity[i]:
                new_word = prefix + c + suffix
                edits.append(new_word)

        # Transposition
        for j in range(i+1, len(entity)):
            mid = entity[i+1:j]
            suffix = entity[j+1:]
            new_word = prefix + entity[j] + mid + entity[i] + suffix
            new_word = new_word.strip()
            if new_word != entity:
                edits.append(new_word)
    return edits


def get_morphological_variants(entity):
    # cooking => cook, cooker
    results = []
    for suffix in ['ing']:
        if entity.endswith(suffix):
            base = entity[:-len(suffix)]
            results.append(base)
            results.append(base + 's')
            results.append(base + 'er')
            results.append(base + 'ers')
    return results

############################################################

class Lexicon(object):
    '''
    A lexicon maps freeform phrases to canonicalized entity.
    The current lexicon just uses several heuristics to do the matching.
    '''
    def __init__(self, schema, learned_lex, word_counts=None):
        start_time = datetime.datetime.now()
        self.schema = schema
        # if True, lexicon uses learned system
        self.learned_lex = learned_lex
        if word_counts:
            self.common_words = [x for x in word_counts if word_counts[x] > 20]
        else:
            self.common_words = []
        self.entities = {}  # Mapping from (canonical) entity to type (assume type is unique)
        self.word_counts = defaultdict(int)  # Counts of words that show up in entities
        self.lexicon = defaultdict(list)  # Mapping from string -> list of (entity, type)
        self.load_entities()
        self.compute_synonyms()
        # TODO: the model cannot handle number entities now
        #self.add_numbers()
        #print 'Ambiguous entities:'
        #for phrase, entities in self.lexicon.items():
        #    if len(entities) > 1:
        #        print phrase, entities

        end_time = datetime.datetime.now()
        print 'Created lexicon in %d seconds: %d phrases mapping to %d entities, %f entities per phrase' % \
              ((end_time - start_time).seconds, len(self.lexicon), len(self.entities),
               sum([len(x) for x in self.lexicon.values()])/float(len(self.lexicon)))

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

    def lookup(self, phrase):
        return self.lexicon.get(phrase, [])

    def compute_synonyms(self):
        # Special cases
        for entity, type in self.entities.items():
            #print entity
            phrases = [entity]  # Representations of the canonical entity
            # Consider any word in the entity that's unique
            # Example: entity = 'university of california', 'university' would not be unique, but 'california' would be
            if ' ' in entity:
                for word in entity.split(' '):
                    if len(word) >= 3 and self.word_counts[word] == 1 and not word.lower() in self.common_words:
                        phrases.append(word)
            # Consider removing stop words
            mod_entity = entity
            for s in [' of ', ' - ']:
                mod_entity = mod_entity.replace(s, ' ')
            if entity != mod_entity:
                phrases.append(mod_entity)

            # Expand!
            synonyms = []
            # Special case
            if entity == 'facebook':
                synonyms.append('fb')
            if type == 'person':
                first_name = entity.split(' ')[0]
                if len(first_name) >= 3 and first_name not in synonyms:
                    synonyms.append(first_name)
            # General
            n = len(entity.split())
            for phrase in phrases:
                synonyms.append(phrase)
                if type != 'person':
                    #synonyms.extend(get_edits(phrase))
                    synonyms.extend(get_morphological_variants(phrase))
                    synonyms.extend(get_prefixes(phrase, common_words=self.common_words))
                    synonyms.extend(get_acronyms(phrase, self.common_words))

            # Add to lexicon
            for synonym in set(synonyms):
                #print synonym, '=>', entity
                self.lexicon[synonym].append((entity, type))

    def add_numbers(self):
        numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        for i, n in enumerate(numbers):
            for phrase in [str(i), n]:
                self.lexicon[phrase].append((str(i), 'number'))

    def link_entity(self, raw_tokens):
        '''
        Add detected entities to each token
        Example: ['i', 'work', 'at', 'apple'] => ['i', 'work', 'at', ('apple', ('apple', 'company'))]
        '''
        i = 0
        entities = []
        while i < len(raw_tokens):
            # Find longest phrase (if any) that matches an entity
            for l in range(5, 0, -1):
                phrase = ' '.join(raw_tokens[i:i+l])
                results = self.lookup(phrase)
                if len(results) > 0:
                    entity = None
                    if self.learned_lex:
                        # Will later use learned system -- returns full candidate set for now
                        entity = results
                    else:
                        # NOTE: if more than one match, use the first one.
                        # TODO: disambiguate
                        # prioritize exact match (e.g. hiking, biking)
                        for result in results:
                            if result[0] == phrase:
                                entity = result
                                break
                        if not entity:
                            entity = results[0]
                    entities.append((phrase, entity))
                    i += l
                    break
            if not results:
                entities.append(raw_tokens[i])
                i += 1
        return entities

    def test(self):
        phrases = ['i', 'physics', 'comp sci', 'econ', 'penn', 'cs', 'upenn', 'u penn', 'u of p', 'ucb', 'berekely', 'jessica']
        phrases = ['foodie', 'evening', 'evenings', 'food']
        for x in phrases:
            print x, '=>', self.lookup(x)
        sentence = 'hiking biking'.split()
        print self.link_entity(sentence)


if __name__ == "__main__":
    from schema import Schema
    from src.model.preprocess import Preprocessor
    from dataset import read_dataset
    from scenario_db import ScenarioDB
    from itertools import chain
    from util import read_json
    import argparse

    #path = 'data/friends-schema-large.json'
    #schema = Schema(path, 'MutualFriends')
    path = 'data/friends-schema.json'
    schema = Schema(path)

    #paths = ['output/friends-scenarios-large.json', 'output/friends-scenarios-large-peaky.json', 'output/friends-scenarios-large-peaky-04-002.json']
    paths = ['output/friends-scenarios.json']
    scenario_db = ScenarioDB.from_dict(schema, (read_json(path) for path in paths))

    #args = {'train_examples_paths': ['data/mutualfriends/train.json'],
    #        'test_examples_paths': ['data/mutualfriends/test.json'],
    #        'train_max_examples': None,
    #        'test_max_examples': None,
    #        }
    args = {'train_examples_paths': ['output/friends-train-examples.json'],
            'test_examples_paths': ['output/friends-test-examples.json'],
            'train_max_examples': None,
            'test_max_examples': None,
            }
    args = argparse.Namespace(**args)
    dataset = read_dataset(scenario_db, args)

    word_counts = Preprocessor.count_words(chain(dataset.train_examples, dataset.test_examples))
    lex = Lexicon(schema, False, word_counts)
    lex.test()
