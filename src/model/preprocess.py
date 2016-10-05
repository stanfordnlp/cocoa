'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from vocab import Vocabulary, is_entity
from graph import Graph

# Special symbols
START = '<d>'
END_TURN = '</t>'
END_UTTERANCE = '</s>'
SELECT = '<select>'
markers = (START, END_TURN, END_UTTERANCE, SELECT)

def tokenize(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    # Split on punctuation
    tokens = re.findall(r"[\w']+|[.,!?;]", utterance)
    return tokens

def build_schema_mappings(schema):
    entity_map = Vocabulary(unk=True)
    for type_, values in schema.values.iteritems():
        entity_map.add_words(((value.lower(), type_) for value in values))

    relation_map = Vocabulary(unk=False)
    attribute_types =  schema.get_attributes()  # {attribute_name: value_type}
    relation_map.add_words((a.lower() for a in attribute_types.keys()))

    return entity_map, relation_map

def build_vocab(examples, lexicon):
    vocab = Vocabulary()
    # Add tokens
    for ex in examples:
        for e in ex.events:
            if e.action == 'message':
                for token in e.processed_data:
                    # Surface form of the entity
                    if is_entity(token):
                        vocab.add_word(token[0])
                    else:
                        vocab.add_word(token)
    # Add entities
    vocab.add_words(lexicon.entities.iteritems())

    # Add special symbols
    vocab.add_words(markers)
    return vocab

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, schema, lexicon, mappings=None, use_kb=False):
        # TODO: change basic/dataset so that it uses the dict structure
        self.examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.num_examples = {k: len(v) if v else 0 for k, v in self.examples.iteritems()}
        self.lexicon = lexicon
        self.use_kb = use_kb
        self.preprocess()
        if not mappings:
            self.vocab = build_vocab(self.examples['train'], lexicon)
            self.entity_map, self.relation_map = build_schema_mappings(schema)
        else:
            self.vocab = mappings['vocab']
            self.entity_map = mappings['entity']
            self.relation_map = mappings['relation']
        print 'Vocabulary size:', self.vocab.size
        print 'Entity size:', self.entity_map.size
        print 'Relation size:', self.relation_map.size

    def preprocess(self):
        for _, examples in self.examples.iteritems():
            if examples:
                for ex in examples:
                    for e in ex.events:
                        if e.action == 'message':
                            # lower, tokenize, link entity
                            entity_tokens = self.lexicon.entitylink(tokenize(e.data))
                            e.processed_data = entity_tokens

    def _get_message_tokens(self, e):
        '''
        Take an event (action and data) and output a sequence of tokens.
        '''
        # NOTE: both inputs and targets are over entities (x[1])
        # TODO: targets over surface tokens (x[0])
        tokens = []
        if e.action == 'message':
            tokens = [x[1] if is_entity(x) else x for x in e.processed_data]
        elif e.action == 'select':
            tokens = [SELECT, (e.data['Name'].lower(), 'person')]
        else:
            raise ValueError('Unknown action.')
        tokens.append(END_UTTERANCE)
        return tokens

    def generator_eval(self, name):
        '''
        Generate vectorized data for testing. Each example is
        a prefix of a dialog. The model should output a response.
        '''
        examples = self.examples[name]
        for ex in examples:
            kbs = ex.scenario.kbs
            prefix = [self.vocab.to_ind(START)]

            for i, e in enumerate(ex.events):
                curr_message = self._get_message_tokens(e)
                if i+1 == len(ex.events) or e.agent != ex.events[i+1].agent:
                    curr_message.append(END_TURN)
                target = map(self.vocab.to_ind, curr_message)

                # This example is considered as a one-turn dialogue during test,
                # therefore we have a new Graph object for the agent
                yield e.agent, kbs[e.agent], \
                    np.array(prefix, dtype=np.int32).reshape(1, -1), \
                    Graph(kbs[e.agent]) if self.use_kb else None, \
                    target  # NOTE: target is a list not numpy array

                prefix.extend(target)

    def generator_train(self, name, shuffle=True):
        '''
        Generate vectorized data for training. Each example is
        a dialog represented as a sequence of tokens. A boolean
        vector `iswrite` indicate whether the agent (0 or 1) is
        speaking.
        '''
        examples = self.examples[name]
        inds = range(len(examples))
        dtype = np.int32

        while True:
            if shuffle:
                random.shuffle(inds)
            for ind in inds:
                ex = examples[ind]
                kbs = ex.scenario.kbs
                # tokens: END_TURN u1 END_UTTERANCE u2 ... END_TURN ...
                tokens = []
                curr_message = [START]  # tokens in one turn
                write = []  # write when agent == 0
                for i, e in enumerate(ex.events):
                    curr_message.extend(self._get_message_tokens(e))
                    if i+1 == len(ex.events) or e.agent != ex.events[i+1].agent:
                        curr_message.append(END_TURN)
                        write.extend([True if e.agent == 0 else False] * len(curr_message))
                        tokens.extend(curr_message)
                        curr_message = []
                token_ids = map(self.vocab.to_ind, tokens)
                n = len(token_ids) - 1
                inputs = np.asarray(token_ids[:-1], dtype=dtype).reshape(1, n)
                targets = np.asarray(token_ids[1:], dtype=dtype).reshape(1, n)
                # write when agent == 0
                yield 0, kbs[0], inputs, Graph(kbs[0]) if self.use_kb else None, targets, np.asarray(write[1:], dtype=np.bool_).reshape(1, n)
                # write when agent == 1
                yield 1, kbs[1], inputs, Graph(kbs[1]) if self.use_kb else None, targets, np.asarray([False if w else True for w in write[1:]], dtype=np.bool_).reshape(1, n)

# test
if __name__ == '__main__':
    import argparse
    from basic.dataset import add_dataset_arguments, read_dataset
    from basic.schema import Schema
    from basic.scenario_db import ScenarioDB, add_scenario_arguments
    from basic.lexicon import Lexicon
    from basic.util import read_json
    from model.kg_embed import add_kg_arguments, CBOWGraph

    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    add_scenario_arguments(parser)
    add_dataset_arguments(parser)
    add_kg_arguments(parser)
    args = parser.parse_args()
    random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    lexicon = Lexicon(schema)
    kg = CBOWGraph(schema, lexicon, args.kg_embed_size, args.rnn_size)

    data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, lexicon)
    data_generator.set_kg(kg)

    gen = data_generator.generator_train('train')
    print '=========== train data ============='
    for i in range(1):
        agent, kb, inputs, entities, targets, iswrite = gen.next()
        print 'INPUTS:\n', inputs, '\n', map(data_generator.vocab.to_word, list(inputs[0]))
        print 'TARGETS:\n', targets, '\n', map(data_generator.vocab.to_word, list(targets[0]))
        print 'WRITE:', iswrite
        print 'ENTITIES:'
        for entity in entities[0]:
            print entity
            print map(lambda x: kg.ind_to_label[x] if x != -1 else None, entity)

    print '=========== eval data ============='
    gen = data_generator.generator_eval('train')
    for i in range(3):
        agent, kb, inputs, entities, targets = gen.next()
        print 'agent=%d' % agent
        print 'INPUTS:\n', inputs, '\n', map(data_generator.vocab.to_word, list(inputs[0]))
        print 'TARGETS:\n', targets, '\n', map(data_generator.vocab.to_word, targets)
        print 'ENTITIES:'
        for entity in entities[0]:
            print entity
            print map(lambda x: kg.ind_to_label[x] if x != -1 else None, entity)

