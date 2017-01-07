import collections
import editdistance
import json
import re
import random

from collections import defaultdict
from fuzzywuzzy import fuzz
from lexicon_utils import get_prefixes, get_acronyms, get_edits, get_morphological_variants


class BaseLexicon(object):
    """
    Base lexicon class defining general purpose functions for any lexicon
    """
    def __init__(self, schema, learned_lex):
        self.schema = schema
        # if True, lexicon uses learned system
        self.learned_lex = learned_lex
        self.entities = {}  # Mapping from (canonical) entity to type (assume type is unique)
        self.word_counts = defaultdict(int)  # Counts of words that show up in entities
        self.lexicon = defaultdict(list)  # Mapping from string -> list of (entity, type)
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
    def __init__(self, schema, learned_lex=False, entity_ranker=None, scenarios_json=None):
        super(Lexicon, self).__init__(schema, learned_lex)
        # TODO: Remove hard-coding (use list of common words/phrases/stop words)
        self.common_phrases = set(["went", "to", "and", "of", "my", "the", "names", "any",
                                   "friends", "at", "for", "in", "many", "partner", "all", "we",
                                   "start", "go", "school", "do", "know", "no", "work"])

        if scenarios_json is not None:
            self._process_kbs(scenarios_json)
        else:
            raise Warning("No scenarios json provided!")

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

            # Multi-token level variants: UPenn, uc berkeley
            if len(mod_entity.split(" ")) > 1:
                phrase_level_prefixes = get_prefixes(mod_entity, min_length=1, max_length=5)
                phrase_level_acronyms = get_acronyms(mod_entity)
                synonyms.extend(phrase_level_acronyms)
                synonyms.extend(phrase_level_prefixes)


            # Add to lexicon
            for synonym in set(synonyms):
                self.lexicon[synonym].append((entity, type))


    def score_and_match(self, span, candidates, agent, uuid, kb_entities):
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
        if not self.learned_lex:
            entity_scores = []
            for c in candidates:
                # Clean up punctuation
                c_s = re.sub("-", " ", c[0])
                span_tokens = span.split()
                entity_tokens = c_s.split()

                if kb_entities is None:
                    continue
                ed = editdistance.eval(span, c[0])
                if c[0] not in kb_entities:
                    # Prioritize exact match
                    if c[0] == span:
                        score = 0
                    else:
                        score = float("inf")
                elif c[0] in kb_entities and span in entity_tokens:
                    score = 0
                # Prioritize multi phrase spans contained in entity
                elif len(span_tokens) > 1 and span in c_s:
                    score = 1
                else:
                    score = ed

                entity_scores.append(c + (score,))

            # Sort entity scores
            entity_scores = sorted(entity_scores, key=lambda x: x[2])

            # If exact match or substring match with an entity
            if entity_scores[0][2] <= 1:
                if span not in self.common_phrases:
                    best_match = entity_scores[0][:2]
                else:
                    best_match = (span, None)
            else:
                best_match = (span, None)
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

    def link_entity(self, raw_tokens, return_entities=False, agent=1, uuid="NONE", kb=None):
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
        # HACK
        assert kb is not None
        kb_entities = kb.entity_set

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
                        best_match = self.score_and_match(phrase, candidate_entities, agent, uuid, kb_entities)
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
                    # else:
                    #     i += l
                    #     entities.append(candidate_entities)
                    #     break

            if not candidate_entities or single_char:
                linked.append(raw_tokens[i])
                i += 1

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



