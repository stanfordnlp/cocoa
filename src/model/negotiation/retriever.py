import os
import time
import shutil
from itertools import izip
from whoosh.fields import SchemaClass, TEXT, STORED, ID
from whoosh import index
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, And
from src.basic.entity import is_entity, Entity, CanonicalEntity
from src.basic.util import read_json
from preprocess import markers
from collections import defaultdict

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
    context = TEXT(stored=True)
    response = STORED

class Retriever(object):
    # NOTE: don't use <> because this is ignored by the analyzer
    START = '_start_'

    def __init__(self, index_dir, index_name='dialogues', context_size=1, rewrite=False, num_candidates=5):
        '''
        Load index from index_dir or build it from dialogues.
        context_size: number of previous utterances to include
        '''
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
        self.parser = QueryParser('context', schema=self.ix.schema, group=OrGroup.factory(0.9))

        self.num_candidates = num_candidates
        self.empty_candidates = [[] for _ in xrange(self.num_candidates)]

        self.search_time = 0.
        self.num_query = 0.
        self.num_empty = 0

    def process_turn(self, turn):
        '''
        Process entities.
        '''
        # Represent price as "[x]" where x is the normalized value
        if len(turn) == 1 and turn[0] == markers.EOS:
            tokens = [self.START]
        else:
            tokens = ['_price_' if is_entity(x) else x for x in turn]
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
        category = dialogue.kb.facts['item']['Category']
        title = dialogue.kb.facts['item']['Title']
        role = dialogue.role
        results = []
        for turn_id, (agent, turn) in enumerate(izip(dialogue.agents, dialogue.token_turns)):
            if agent != dialogue.agent:
                candidates = None
            else:
                candidates = self.search(role, category, title, prev_turns)
            if json_dict:
                r = {
                        'uuid': dialogue.uuid,
                        'role': dialogue.role,
                        'kb': dialogue.kb.to_dict(),
                        'agent': agent,
                        'turn_id': turn_id,
                        'prev_turns': prev_turns,
                        'target': turn,
                        'candidates': candidates,
                        }
            else:
                r = candidates
            results.append(r)
            prev_turns.append(turn)
        return results

    def dialogue_to_docs(self, d, context_size):
        '''
        Convert a dialogue to docs accoring to the schema.
        '''
        assert d.flattened
        assert len(d.agents) == len(d.token_turns)
        context = []
        docs = []
        role = d.role
        for agent, turn in izip(d.agents, d.token_turns):
            text = self.process_turn(turn)
            #if len(context) > 0:
            # NOTE: the dialogue is from one agent's perspective
            if agent == d.agent:
                doc = {
                        'role': unicode(role),
                        'category': unicode(d.kb.facts['item']['Category']),
                        'title': unicode(d.kb.facts['item']['Title']),
                        'context': unicode(' '.join(context[-1*context_size:])),
                        'response': turn,
                        }
                #print '---------------------'
                #for k, v in doc.iteritems():
                #    print k
                #    print v
                #print '---------------------'
                docs.append(doc)
            context.append(text)
        return docs

    def build_index(self, dialogues):
        writer = self.ix.writer()
        for d in dialogues:
            docs = self.dialogue_to_docs(d, self.context_size)
            for doc in docs:
                writer.add_document(**doc)
        writer.commit()

    def get_query(self, context):
        context = unicode(' '.join([self.process_turn(t) for t in context]))
        query = self.parser.parse(context)
        return query

    def search(self, role, category, title, prev_turns):
        query = self.get_query(prev_turns[-1*self.context_size:])
        # Only consider buyer/seller utterances
        filter_query = And([Term('role', unicode(role)), Term('category', unicode(category))])
        start_time = time.time()
        with self.ix.searcher() as searcher:
            results = searcher.search(query, filter=filter_query, limit=self.num_candidates, terms=True)
            # One more try
            if len(results) == 0:
                query = self.get_query(prev_turns[-1*(self.context_size+1):])
                results = searcher.search(query, filter=filter_query, limit=self.num_candidates, terms=True)

            results = [{'response': r['response'],
                        'context': r['context'],
                        'hits': [x[1] for x in r.matched_terms()],
                        } for r in results]
        n = len(results)
        if n == 0:
            self.num_empty += 1
        if n < self.num_candidates:
            results.extend([{} for _ in xrange(self.num_candidates - n)])
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
        to_ent = lambda x: x if isinstance(x, basestring) else Entity(x[0], CanonicalEntity(*x[1]))
        for path in paths:
            print 'Load candidates from', path
            results = read_json(path)
            for r in results:
                # r['candidates']: num_candidates responses for each turn
                if r['candidates'] is not None:
                    for c in r['candidates']:
                        if 'response' in c:
                            c['response'] = [to_ent(x) for x in c['response']]
                candidates[(r['uuid'], r['role'])].append(r['candidates'])
        return candidates


if __name__ == '__main__':
    from src.basic.negotiation.price_tracker import PriceTracker
    from src.basic.dataset import read_dataset, add_dataset_arguments
    from src.basic.scenario_db import add_scenario_arguments
    from src.basic.schema import Schema
    from src.model.negotiation.preprocess import Preprocessor, markers
    from src.basic.util import write_json
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--retriever-output', help='Output path of json file containing retrieved candidates of test examples')
    add_dataset_arguments(parser)
    add_retriever_arguments(parser)
    args = parser.parse_args()

    dataset = read_dataset(None, args)
    lexicon = PriceTracker()
    schema = Schema(args.schema_path)

    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')

    retriever = Retriever(args.index, context_size=args.retriever_context_len, rewrite=args.rewrite_index, num_candidates=args.num_candidates)
    if not retriever.loaded_index:
        dialogues = preprocessor.preprocess(dataset.train_examples)
        print 'Building index from %s' % ','.join(args.train_examples_paths)
        start_time = time.time()
        retriever.build_index(dialogues)
        print '[%d s]' % (time.time() - start_time)

    # Write a json file of all the candidates
    if args.test_examples_paths and args.retriever_output:
        dialogues = preprocessor.preprocess(dataset.test_examples)
        print 'Retrieving candidates for %s' % ','.join(args.test_examples_paths)
        start_time = time.time()
        results = []
        for dialogue in dialogues:
            results.extend(retriever.retrieve_candidates(dialogue, json_dict=True))
        print '[%d s]' % (time.time() - start_time)
        write_json(results, args.retriever_output)

    #prev_turns = ["</s>".split()]
    #results = retriever.search('buyer', 'bike', '', prev_turns)
    #for r in results:
    #    print r
