import os
import time
import shutil
import random
from itertools import izip, ifilter
from collections import defaultdict
from whoosh.fields import SchemaClass, TEXT, STORED, ID, NUMERIC
from whoosh import index
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, And, Phrase, Or
from cocoa.core.entity import is_entity, Entity, CanonicalEntity
from cocoa.core.util import read_json, generate_uuid
from cocoa.lib.bleu import compute_bleu
from preprocess import markers

def add_retriever_arguments(parser):
    parser.add_argument('--retrieve', action='store_true', help='Use retrieval-based method')
    parser.add_argument('--retriever-context-len', default=1, type=int, help='Number of previous turns to be used as context for retrieval')
    parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates to return')
    parser.add_argument('--index', help='Path to index directory')
    parser.add_argument('--rewrite-index', action='store_true', help='Rewrite the saved index')

class DialogueSchema(SchemaClass):
    role = ID #(stored=True)
    category = ID #(stored=True)
    title = TEXT
    pos = NUMERIC(stored=True)
    prev_context = TEXT
    immediate_context = TEXT(stored=True)
    response = STORED

class Retriever(object):
    def __init__(self, index_dir, index_name='dialogues', context_size=1, rewrite=False, num_candidates=5, slot_detector=None):
        '''
        Load index from index_dir or build it from dialogues.
        context_size: number of previous utterances to include
        '''
        self.slot_detector = slot_detector
        self.index_name = index_name
        if not index.exists_in(index_dir, indexname=index_name) or rewrite:
            if not os.path.exists(index_dir):
                print 'Create index in', index_dir
                os.makedirs(index_dir)
            elif rewrite:
                print 'Rewrite index in', index_dir
                shutil.rmtree(index_dir)
                os.makedirs(index_dir)
            self.ix = index.create_in(index_dir, schema=DialogueSchema, indexname=index_name)
            self.loaded_index = False
        else:
            print 'Load index from', index_dir
            self.ix = index.open_dir(index_dir, indexname=index_name)
            self.loaded_index = True
        self.context_size = context_size
        self.parser_icontext = QueryParser('immediate_context', schema=self.ix.schema, group=OrGroup.factory(0.9))
        self.parser_pcontext  = QueryParser('prev_context', schema=self.ix.schema, group=OrGroup.factory(0.9))
        self.parser_title = QueryParser('title', schema=self.ix.schema, group=OrGroup.factory(0.9))

        self.num_candidates = num_candidates
        self.empty_candidates = [{} for _ in xrange(self.num_candidates)]

        self.search_time = 0.
        self.num_query = 0.
        self.num_empty = 0

    @classmethod
    def process_turn(cls, turn):
        '''
        Process entities.
        '''
        # Represent price as "[x]" where x is the normalized value
        if len(turn) == 1 and turn[0] == markers.EOS:
            # NOTE: don't use <> because this is ignored by the analyzer
            tokens = ['_start_']
        else:
            tokens = ['_price_' if is_entity(x) else x for x in turn]
            tokens = tokens
        return ' '.join(tokens)

    def retrieve_candidates(self, dialogue, json_dict=False):
        '''
        dialogue: a Dialogue object
        json_dict: if True, return a list of dictionary containing kb, context etc.;
            otherwise just a list of candidates corresponding to each turn.
            NOTE: candidates are only available to 'agent speaking' turns.
        return a candidate list for each 'decoding' turn.
        '''
        prev_turns = []
        prev_roles = []
        category = dialogue.kb.facts['item']['Category']
        title = dialogue.kb.facts['item']['Title']
        role = dialogue.role
        results = []
        for turn_id, (agent, turn, role) in enumerate(izip(dialogue.agents, dialogue.token_turns, dialogue.roles)):
            if agent != dialogue.agent:
                candidates = None
            else:
                candidates = self.search(role, category, title, prev_turns)
            if json_dict:
                r = {
                        'exid': generate_uuid('E'),
                        'uuid': dialogue.uuid,
                        'role': dialogue.role,
                        'kb': dialogue.kb.to_dict(),
                        'agent': agent,
                        'turn_id': turn_id,
                        'prev_turns': list(prev_turns),
                        'prev_roles': list(prev_roles),
                        'target': turn,
                        'candidates': candidates,
                        }
                if self.slot_detector:
                    r['kb_context'] = self.slot_detector.get_context_tokens(dialogue.kb)
            else:
                r = candidates
            results.append(r)
            prev_turns.append(turn)
            prev_roles.append(role)
        return results

    def rewrite_candidates(self, candidates, fillers):
        new_candidates = list(candidates)

        def has_slot(tokens):
            for tok in tokens:
                if is_entity(tok) and tok.canonical.type == 'slot':
                    return True
            return False

        for tokens in candidates:
            if has_slot(tokens):
                new_candidates.extend(self.fill_slots(tokens, fillers))


    def dialogue_to_docs(self, d, context_size, written):
        '''
        Convert a dialogue to docs accoring to the schema.
        '''
        assert len(d.agents) == len(d.token_turns)
        context = []
        docs = []
        role = d.role
        L = float(len(d.token_turns)) - 1
        if self.slot_detector:
            kb_context = self.slot_detector.get_context_tokens((d.kb,))
        for i, (agent, turn) in enumerate(izip(d.agents, d.token_turns)):
            text = self.process_turn(turn)
            # NOTE: the dialogue is from one agent's perspective
            if agent == d.agent:
                doc = {
                        'role': unicode(role),
                        'category': unicode(d.kb.facts['item']['Category']),
                        'title': unicode(d.kb.facts['item']['Title']),
                        'pos': i / L,
                        'immediate_context': unicode(context[-1], 'utf8'),
                        'prev_context': unicode(' '.join(context[-1*context_size:-1]), 'utf8'),
                        'response': turn,
                        }
                if self.slot_detector:
                    doc['response'] = self.slot_detector.detect_slots(turn, d.kb, context=kb_context)
                docs.append(doc)
            context.append(text)
        return docs

    def build_index(self, dialogues):
        writer = self.ix.writer()
        written = set()
        for d in dialogues:
            docs = self.dialogue_to_docs(d, self.context_size, written)
            for doc in docs:
                writer.add_document(**doc)
        writer.commit()

    def get_query(self, context, title):
        turns = [self.process_turn(t) for t in context]
        q1 = self.parser_icontext.parse(unicode(turns[-1], 'utf8'))
        q2 = self.parser_pcontext.parse(unicode(' '.join(turns[:-1]), 'utf8'))
        q3 = self.parser_title.parse(unicode(' '.join(title)))
        terms = list(q1.all_terms()) + list(q2.all_terms()) + list(q3.all_terms())
        query = Or([Term(*x) for x in terms])
        return query

    def remove_duplicates(self, results):
        response_set = set()
        no_duplicate_results = []
        for r in results:
            # only check short utterances
            if len(r) < 10:
                key = tuple([x if not is_entity(x) else x.canonical.type for x in r['response'] if x not in ('.', ',', '!', '?')])
                if key not in response_set:
                    no_duplicate_results.append(r)
                response_set.add(key)
            else:
                no_duplicate_results.append(r)
        return no_duplicate_results

    def search(self, role, category, title, prev_turns):
        query = self.get_query(prev_turns[-1*self.context_size:], title)
        # Only consider buyer/seller utterances
        filter_query = And([Term('role', unicode(role)), Term('category', unicode(category))])
        start_time = time.time()
        with self.ix.searcher() as searcher:
            results = searcher.search(query, filter=filter_query, limit=self.num_candidates, terms=True)
            # One more try
            if len(results) == 0:
                query = self.get_query(prev_turns[-1*(self.context_size+1):], title)
                results = searcher.search(query, filter=filter_query, limit=self.num_candidates, terms=True)

            results = self.remove_duplicates(results)
            results = [{'response': r['response'],
                        'context': r['immediate_context'],
                        'hits': [x[1] for x in r.matched_terms()],
                        'pos': r['pos'],
                        } for r in results]

        # Sort by BLEU
        ref = self.process_turn(prev_turns[-1]).split()
        results = sorted(results, key=lambda r: compute_bleu(r['context'], ref), reverse=True)

        offered = markers.OFFER in prev_turns[-1]
        if not offered:
            results = [r for r in results if not
                    (markers.ACCEPT in r['response'] or markers.REJECT in r['response'])]
        else:
            results = [r for r in results if
                    (markers.ACCEPT in r['response'] or markers.REJECT in r['response'])]
            if len(results) == 0:
                results.append({'response': [markers.ACCEPT], 'context': [], 'hits': []})
                results.append({'response': [markers.REJECT], 'context': [], 'hits': []})

        n = len(results)
        if n == 0:
            self.num_empty += 1

        #if n < self.num_candidates:
        #    results.extend([{} for _ in xrange(self.num_candidates - n)])

        self.num_query += 1
        self.search_time += (time.time() - start_time)
        return results

    def report_search_time(self):
        if self.num_query == 0:
            time = -1
        else:
            time = self.search_time / self.num_query
        print 'Average search time per query: [%f s]' % time
        print 'Number of empty result:', self.num_empty

    def load_candidates(self, paths):
        candidates = defaultdict(list)
        # When dumped to json, NamedTuple becomes list. Now convert it back.
        is_str = lambda x: isinstance(x, basestring)
        # x[0] (surface of entity): note that for prices from the offer action,
        # surface is float instead of string
        to_ent = lambda x: x.encode('utf-8') if is_str(x) else \
            Entity(x[0].encode('utf-8') if is_str(x[0]) else x[0], CanonicalEntity(*x[1]))
        for path in paths:
            print 'Load candidates from', path
            results = read_json(path)
            for r in results:
                # None for encoding turns
                if r['candidates'] is None:
                    candidates[(r['uuid'], r['role'])].append(None)
                else:
                    # Only take the response (list of tokens)
                    candidates_ = [[to_ent(x) for x in c['response']]
                        for c in ifilter(lambda x: 'response' in x, r['candidates'])]
                    candidates[(r['uuid'], r['role'])].append(candidates_)
        return candidates

if __name__ == '__main__':
    import argparse
    import time

    from cocoa.core.dataset import read_dataset, add_dataset_arguments
    from cocoa.core.schema import Schema
    from cocoa.core.util import write_json

    from core.price_tracker import PriceTracker, add_price_tracker_arguments
    from core.slot_detector import SlotDetector, add_slot_detector_arguments
    from core.scenario import Scenario

    from preprocess import Preprocessor, markers

    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--retriever-output', help='Output path of json file containing retrieved candidates of test examples')
    parser.add_argument('--verbose', action='store_true', help='Print retrieved candidates')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for generating exid')
    add_dataset_arguments(parser)
    add_retriever_arguments(parser)
    add_price_tracker_arguments(parser)
    add_slot_detector_arguments(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset = read_dataset(args, Scenario)
    lexicon = PriceTracker(args.price_tracker_model)
    if args.slot_scores is not None:
        slot_detector = SlotDetector(slot_scores_path=args.slot_scores)
    else:
        slot_detector = None
    schema = Schema(args.schema_path)

    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')

    #dialogues = preprocessor.preprocess(dataset.train_examples)
    #uuids = set()
    #for d in dialogues:
    #    if d.uuid in uuids:
    #        continue
    #    uuids.add(d.uuid)
    #    for turn in d.token_turns:
    #        s = ['[%.2f]' % x.canonical.value if is_entity(x) else x for x in turn if x != markers.EOS]
    #        if len(s) > 0:
    #            print ' '.join(s)
    #import sys; sys.exit()

    retriever = Retriever(args.index, context_size=args.retriever_context_len, rewrite=args.rewrite_index, num_candidates=args.num_candidates, slot_detector=slot_detector)

    if not retriever.loaded_index:
        dialogues = preprocessor.preprocess(dataset.train_examples)
        print 'Building index from %s' % ','.join(args.train_examples_paths)
        start_time = time.time()
        retriever.build_index(dialogues)
        print '[%d s]' % (time.time() - start_time)

    def to_str(tokens):
        return ' '.join([tok if not is_entity(tok) else '[%s]' % tok.surface for tok in tokens])

    def dump_result(result):
        if result['candidates'] is None:
            return
        print '================='
        print 'ROLE:', result['role']
        print 'CATEGORY:', result['kb']['item']['Category']
        print 'TITLE:', result['kb']['item']['Title']
        print 'CONTEXT:'
        for role, turn in izip(result['prev_roles'], result['prev_turns']):
            print '%s: %s' % (role, to_str(turn))
        print 'TARGET:'
        print to_str(result['target'])
        print 'KB CONTEXT:', result.get('kb_context', None)
        print 'CANDIDATES:'
        for c in result['candidates']:
            if 'response' in c:
                print '----------'
                print c['hits']
                print to_str(c['response'])
                #print c['context']
                #print c['pos']

    # Write a json file of all the candidates
    if args.test_examples_paths and args.retriever_output:
        dialogues = preprocessor.preprocess(dataset.test_examples)
        print 'Retrieving candidates for %s' % ','.join(args.test_examples_paths)
        start_time = time.time()
        results = []
        for dialogue in dialogues:
            result = retriever.retrieve_candidates(dialogue, json_dict=True)
            results.extend(result)
            if args.verbose:
                for r in result:
                    dump_result(r)
        print '[%d s]' % (time.time() - start_time)
        write_json(results, args.retriever_output)

    #prev_turns = ["I 'm a poor student . can you go lower ?".split()]
    #results = retriever.search('seller', 'furniture', '', prev_turns)
    #for r in results:
    #    print r
