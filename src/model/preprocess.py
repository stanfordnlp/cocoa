'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from vocab import Vocabulary

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

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, lexicon, vocab=None):
        self.examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.num_examples = {k: len(v) if v else 0 for k, v in self.examples.iteritems()}
        self.lexicon = lexicon
        # If the generator does not interact with a KB, we don't need to get entities
        self.get_entity_list = lambda *args: None
        self.preprocess()
        if not vocab:
            self.vocab = Vocabulary()
            self.build_vocab()
        else:
            print 'Loaded vocabulary size:', vocab.size
            self.vocab = vocab

    def set_kg(self, kg):
        self.kg = kg
        self.get_entity_list = self.kg.get_entity_list

    def preprocess(self):
        for _, examples in self.examples.iteritems():
            if examples:
                for ex in examples:
                    for e in ex.events:
                        if e.action == 'message':
                            # lower, tokenize, link entity
                            entity_tokens = self.lexicon.entitylink(tokenize(e.data))
                            e.processed_data = entity_tokens

    def build_vocab(self):
        # Add tokens
        for ex in self.examples['train']:
            for e in ex.events:
                if e.action == 'message':
                    for token in e.processed_data:
                        if isinstance(token, basestring):
                            self.vocab.add_words(token)
                        else:
                            # Surface form of the entity
                            self.vocab.add_words(token[0])

        # Add entities
        self.vocab.add_words(self.lexicon.entities.iteritems())

        # Add special symbols
        self.vocab.add_words(markers)

        self.vocab.build()

    def _get_message_tokens(self, e):
        '''
        Take an event (action and data) and output a sequence of tokens.
        '''
        # NOTE: both inputs and targets are over entities (x[1])
        # TODO: targets over surface tokens (x[0])
        tokens = []
        if e.action == 'message':
            tokens = [x if isinstance(x, basestring) else x[1] for x in e.processed_data]
        elif e.action == 'select':
            tokens = [SELECT, (e.data['Name'].lower(), 'person')]
        else:
            raise ValueError('Unknown action.')
        tokens.append(END_UTTERANCE)
        return tokens

    def entity_list_to_array(self, entities):
        '''
        Transform entity list from kg_embed to numpy array that's fed to tf models.
        '''
        if entities is None:
            return None
        # Require different shapes because the tf graph will be different
        if not self.kg.train_utterance:
            return np.asarray(entities, dtype=np.int32).reshape(1, -1, self.kg.entity_cache_size)
        else:
            return np.asarray(entities, dtype=np.int32).reshape(1, -1, self.kg.entity_cache_size, self.kg.utterance_size, 2)

    def generator_eval(self, name):
        '''
        Generate vectorized data for testing. Each example is
        a prefix of a dialog. The model should output a response.
        '''
        examples = self.examples[name]
        for ex in examples:
            kbs = ex.scenario.kbs
            prefix = [self.vocab.to_ind(START)]
            entities = self.get_entity_list([START])
            for i, e in enumerate(ex.events):
                curr_message = self._get_message_tokens(e)
                if i+1 == len(ex.events) or e.agent != ex.events[i+1].agent:
                    curr_message.append(END_TURN)
                curr_entities = self.get_entity_list(curr_message)
                curr_message = map(self.vocab.to_ind, curr_message)
                yield e.agent, kbs[e.agent], \
                    np.array(prefix, dtype=np.int32).reshape(1, -1), \
                    self.entity_list_to_array(entities), \
                    curr_message  # NOTE: target is a list not numpy array
                prefix.extend(curr_message)
                if entities is not None:
                    entities.extend(curr_entities)

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
                entities = self.get_entity_list(tokens)
                tokens = map(self.vocab.to_ind, tokens)
                n = len(tokens) - 1
                inputs = np.asarray(tokens[:-1], dtype=dtype).reshape(1, n)
                if entities:
                    entities = self.entity_list_to_array(entities[:-1])
                targets = np.asarray(tokens[1:], dtype=dtype).reshape(1, n)
                # write when agent == 0
                yield 0, kbs[0], inputs, entities, targets, np.asarray(write[1:], dtype=np.bool_).reshape(1, n)
                # write when agent == 1
                yield 1, kbs[1], inputs, entities, targets, np.asarray([False if w else True for w in write[1:]], dtype=np.bool_).reshape(1, n)

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

