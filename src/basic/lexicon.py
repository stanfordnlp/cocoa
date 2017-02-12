import collections
import editdistance
import json
import re
import random

from collections import defaultdict
from fuzzywuzzy import fuzz
from lexicon_utils import get_prefixes, get_acronyms, get_edits, get_morphological_variants

def add_lexicon_arguments(parser):
    parser.add_argument('--stop-words', type=str, default='data/common_words.txt', help='Path to stop words list')
    parser.add_argument('--learned-lex', default=False, action='store_true', help='if true have entity linking in lexicon use learned system')
    parser.add_argument('--inverse-lexicon', help='Path to inverse lexicon data')

class BaseLexicon(object):
    """
    Base lexicon class defining general purpose functions for any lexicon
    """
    def __init__(self, schema, learned_lex, stop_words=None):
        self.schema = schema
        # if True, lexicon uses learned system
        self.learned_lex = learned_lex
        self.entities = {}  # Mapping from (canonical) entity to type (assume type is unique)
        self.word_counts = defaultdict(int)  # Counts of words that show up in entities
        self.lexicon = defaultdict(list)  # Mapping from string -> list of (entity, type)
        with open(stop_words, 'r') as fin:
            self.stop_words = set([x.strip() for x in fin.read().split()][:1000])
            self.stop_words.update(['one', '1', 'two', '2', 'three', '3', 'four', '4', 'five', '5', 'six', '6', 'seven', '7', 'eight', '8', 'nine', '9', 'ten', '10'])
        self.load_entities()
        self.compute_synonyms()
        print 'Created lexicon: %d phrases mapping to %d entities, %f entities per phrase' % (len(self.lexicon), len(self.entities), sum([len(x) for x in self.lexicon.values()])/float(len(self.lexicon)))


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



class Lexicon(BaseLexicon):
    """
    Lexicon that only computes per token entity transforms rather than per phrase transforms (except for prefixes/acronyms)
    """
    def __init__(self, schema, learned_lex=False, entity_ranker=None, scenarios_json=None, stop_words=None):
        super(Lexicon, self).__init__(schema, learned_lex, stop_words)
        # TODO: Remove hard-coding (use list of common words/phrases/stop words)
        self.common_phrases = set(["went", "to", "and", "of", "my", "the", "names", "any",
                                   "friends", "at", "for", "in", "many", "partner", "all", "we",
                                   "start", "go", "school", "do", "know", "no", "work", "are",
                                   "he", "she"])

        # Ensure an entity ranker is provided for scoring (span, entity) pairs
        if learned_lex:
            assert entity_ranker is not None
            print "Using learned lexicon..."
            self.entity_ranker = entity_ranker
        else:
            print "Using rule-based lexicon..."


    def _process_kbs(self, scenarios_json):
        """
        Process kb scenarios
        :param scenarios_json: Path to scenarios json file
        :return:
        """
        with open(scenarios_json, "r") as f:
            scenarios_info = json.load(f)

        # Map from uuid to KBs
        uuid_to_kbs_with_types = collections.defaultdict(dict)
        uuid_to_kbs = collections.defaultdict(dict)
        for scenario in scenarios_info:
            uuid = scenario["uuid"]
            # Keep track of separate mappings to entities with types and entities without types
            agent_kbs = {0: set(), 1: set()}
            agent_kbs_with_types = {0: set(), 1: set()}

            for agent_idx, kb in enumerate(scenario["kbs"]):
                for item in kb:
                    row_entities = item.items()
                    row_entities_with_types = [(e[0], e[1].lower()) for e in row_entities]
                    row_entities = [e[1].lower() for e in row_entities]
                    agent_kbs_with_types[agent_idx].update(row_entities_with_types)
                    agent_kbs[agent_idx].update(row_entities)

            uuid_to_kbs_with_types[uuid] = agent_kbs_with_types
            uuid_to_kbs[uuid] = agent_kbs


        self.uuid_to_kbs = uuid_to_kbs
        self.uuid_to_kbs_with_types = uuid_to_kbs_with_types


    def compute_synonyms(self):
        """
        Computes all variants (synonyms) for each token of every canonical entity
        :return:
        """
        # Keep track of tokens we have seen to handle repeats
        for entity, type in self.entities.items():
            phrases = []
            mod_entity = entity
            for s in [' of ', ' - ', '-']:
                mod_entity = mod_entity.replace(s, ' ')

            # Add all tokens in entity -- we only compute token-level edits (except for acronyms/prefixes...)
            entity_tokens = mod_entity.split(' ')
            phrases.extend([t for t in entity_tokens])

            synonyms = []
            if entity == 'facebook':
                synonyms.append('fb')

            # General
            for phrase in phrases:
                synonyms.append(phrase)
                if type != 'person':
                    synonyms.extend(get_edits(phrase))
                    synonyms.extend(get_morphological_variants(phrase))
                    synonyms.extend(get_prefixes(phrase, min_length=1))
                if phrase in ('and', '&', "'n"):
                    synonyms.extend(['and', '&', "'n"])

            # Multi-token level variants: UPenn, uc berkeley
            if len(mod_entity.split(" ")) > 1:
                phrase_level_prefixes = get_prefixes(mod_entity, min_length=1, max_length=5)
                phrase_level_acronyms = get_acronyms(mod_entity)
                synonyms.extend(phrase_level_acronyms)
                synonyms.extend(phrase_level_prefixes)


            # Add to lexicon
            for synonym in set(synonyms):
                if self.stop_words and synonym not in self.word_counts and synonym in self.stop_words:
                    continue
                self.lexicon[synonym].append((entity, type))

    def score_and_match(self, span, candidates, agent, uuid, kb_entities, kb_entity_types, known_kb=True):
        """
        Score the given span with the list of candidate entities and returns best match
        :param span:
        :param candidates:
        :param kb_entities: Set of entities mentioned in both agents KBs
        :param agent: Agent id whose span is being entity linked
        :param uuid: uuid of scenario containing KB for given agent
        :return:
        """
        # Use heuristic scoring system
        #print 'span:', span
        if not self.learned_lex:
            entity_scores = []
            for c in candidates:
                #print 'c:', c
                # Clean up punctuation
                c_s = re.sub("-", " ", c[0])
                span_tokens = span.split()
                entity_tokens = c_s.split()

                ed = editdistance.eval(span, c[0])
                # Filter false positives
                if c[1] not in kb_entity_types:
                    #print 'false type'
                    continue

                def is_stopwords():
                    if span == c[0]:
                        return False
                    if len(span_tokens) == 1 and span in self.stop_words:
                        return True
                    if span_tokens[0] in ('and', 'or', 'to', 'from', 'of', 'in', 'at'):
                        return True
                    all_stop = True
                    for x in span_tokens:
                        if x not in self.stop_words:
                            all_stop = False
                            break
                    if all_stop:
                        return True
                    return False

                if is_stopwords():
                    #print 'stop words'
                    continue
                if len(span_tokens) > len(entity_tokens):
                    continue
                if c[0] not in kb_entities and known_kb:
                    # Prioritize exact match
                    if c[0] == span:
                        score = 0
                    else:
                        #print 'not in kb'
                        continue
                elif span in entity_tokens:
                    score = 0
                # Prioritize multi phrase spans contained in entity
                elif len(span_tokens) > 1 and span in c_s:
                    score = 1
                else:
                    score = ed + 2
                # Prioritize entity in KB even if we are not sure
                if not known_kb and c[0] not in kb_entities and c[0] != span:
                    score += 3
                #print 'score:', score

                entity_scores.append(c + (score,))

            # Sort entity scores
            if len(entity_scores) == 0:
                return (span, None)
            entity_scores = sorted(entity_scores, key=lambda x: x[2])

            # If exact match or substring match with an entity
            entity, type_, score = entity_scores[0]

            # Be more cautious when not known_kb; +3 because previous prioritization
            if score > 8 and not known_kb:
                best_match = (span, None)
            elif (score > 5 and len(entity_scores) > 1) or span in self.common_phrases:
                best_match = (span, None)
            else:
                best_match = (entity, type_)
        else:
            # Use learned ranker
            entity_scores = []
            for c in candidates:
                score = self.entity_ranker.score(span, c[0], agent, uuid).squeeze()
                entity_scores.append(c + (score[0] - score[1],))

            # Where does original span fit into all this? If smaller than some threshold
            span_score = self.entity_ranker.score(span, span, agent, uuid).squeeze()

            # Sort entity scores
            entity_scores = sorted(entity_scores, key=lambda x: x[2])
            best_entity = entity_scores[0][:3]

            if (span_score[0] - span_score[1]) < best_entity[2]:
                best_match = (span, None)
            else:
                best_match = best_entity[:2]

        return best_match

    def combine_repeated_entity(self, entity_tokens):
        '''
        Remove false positives when a span for one entity is recognized multiple times.
        '''
        is_entity = lambda x: not isinstance(x, basestring)
        prev_entity = None
        max_dist = 1
        cache = []
        combined_entity_tokens = []
        for i, token in enumerate(entity_tokens):
            if is_entity(token):
                if prev_entity is not None and token[0] != prev_entity[0] and token[1] == prev_entity[1] and (len(cache) <= max_dist):
                    surface = '%s %s %s' % (prev_entity[0], ' '.join(cache), token[0])
                    combined_entity_tokens[-1] = (surface, prev_entity[1])
                else:
                    combined_entity_tokens.extend(cache)
                    combined_entity_tokens.append(token)
                prev_entity = token
                cache = []
            elif prev_entity is None:
                combined_entity_tokens.append(token)
            else:
                cache.append(token)
        combined_entity_tokens.extend(cache)
        return combined_entity_tokens

    def link_entity(self, raw_tokens, return_entities=False, agent=1, uuid="NONE", kb=None, mentioned_entities=None, known_kb=True):
        """
        Add detected entities to each token
        Example: ['i', 'work', 'at', 'apple'] => ['i', 'work', 'at', ('apple', ('apple','company'))]
        Note: Linking works differently here because we are considering intersection of lists across
        token spans so that "univ of penn" will lookup in our lexicon table for "univ" and "penn"
        (disregarding stop words and special tokens) and find their intersection
        :param return_entities: Whether to return entities found in utterance
        :param agent: Agent (0,1) whose utterance is being linked
        :param uuid: uuid of scenario being used for testing whether candidate entity is in KB
        """
        if kb is not None:
            kb_entities = kb.entity_set
            if mentioned_entities is not None:
                kb_entities = kb_entities.union(mentioned_entities)
            kb_entity_types = kb.entity_type_set
        else:
            kb_entities = None
            kb_entity_types = None

        i = 0
        found_entities = []
        linked = []
        stop_words = set(['of'])
        while i < len(raw_tokens):
            candidate_entities = None
            single_char = False
            # Find longest phrase (if any) that matches an entity
            for l in range(6, 0, -1):
                phrase = ' '.join(raw_tokens[i:i+l])
                raw = raw_tokens[i:i+l]

                for idx, token in enumerate(raw):
                    results = self.lookup(token)
                    if idx == 0: candidate_entities = results
                    if token not in stop_words:
                        candidate_entities = list(set(candidate_entities).intersection(set(results)))

                # Single character token so disregard candidate entities
                if l == 1 and len(phrase) == 1:
                    single_char = True
                    break

                # Found some match
                if len(candidate_entities) > 0:
                    if kb_entities is not None:
                        best_match = self.score_and_match(phrase, candidate_entities, agent, uuid, kb_entities, kb_entity_types, known_kb)
                    else:
                        # TODO: Fix default system, if no kb_entities provided -- only returns random candidate now
                        best_match = random.sample(candidate_entities, 1)[0]
                    # If best_match is entity from KB add to list
                    if best_match[1] is not None:
                        # Return as (surface form, (canonical, type))
                        linked.append((phrase, best_match))
                        found_entities.append((phrase, best_match))
                        i += l
                        break
                    else:
                        candidate_entities = None
                        continue

            if not candidate_entities or single_char:
                linked.append(raw_tokens[i])
                i += 1

        linked = self.combine_repeated_entity(linked)

        # For computing per dialogue entities found
        if return_entities:
            return linked, found_entities

        return linked


    def test(self):
        sentence3 = "I went to University of Pensylvania and most my friends are from there".split(" ")
        sentence3 = "cal"#"from Cal State Chico"
        sentence3 = [t.lower() for t in sentence3.split()]

        sentence2 = ["zach"]
        print self.link_entity(sentence3, True, 1, "S_cxqu6PM56ACAiDLi")
        #print self.link_entity(sentence2, True)
        #print get_prefixes("biology")


if __name__ == "__main__":
    from schema import Schema
    import argparse
    import time
    from entity_ranker import EntityRanker

    parser = argparse.ArgumentParser("arguments for basic testing lexicon")
    parser.add_argument("--schema", type=str, help="path to schema to use")
    parser.add_argument("--ranker-data", type=str, help="path to train data")
    parser.add_argument("--annotated-examples-path", help="Json of annotated examples", type=str)
    parser.add_argument("--scenarios-json", help="Json of scenario information", type=str)
    parser.add_argument("--transcripts", help="Json file of all transcripts collected")

    args = parser.parse_args()

    path = args.schema
    start_build = time.time()

    ranker = EntityRanker(args.annotated_examples_path, args.scenarios_json, args.ranker_data, args.transcripts)
    schema = Schema(path)
    lex = Lexicon(schema, learned_lex=True, entity_ranker=ranker, scenarios_json=args.scenarios_json)
    print "Building complete: ", time.time() - start_build
    start_test = time.time()
    lex.test()
    print "Testing Complete: ", time.time() - start_test



