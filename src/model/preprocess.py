'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from vocab import Vocabulary

# Special symbols
END_TURN = '</t>'
END_UTTERANCE = '</s>'
SELECT = '<select>'
markers = (END_TURN, END_UTTERANCE, SELECT)

def tokenize(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    # Split on punctuation
    tokens = re.findall(r"[\w']+|[.,!?;]", utterance)
    return tokens

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, lexicon):
        self.examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.lexicon = lexicon
        self.preprocess()
        self.vocab = Vocabulary()
        self.build_vocab()

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

    def generator(self, name):
        '''
        Generate vectorized data for NN training
        '''
        examples = self.examples[name]
        inds = range(len(examples))
        dtype = 'int32'

        while True:
            random.shuffle(inds)
            for ind in inds:
                ex = examples[ind]
                # tokens: END_TURN u1 END_UTTERANCE u2 ... END_TURN ...
                tokens = []
                curr_agent = -1
                for e in ex.events:
                    if e.agent != curr_agent:
                        tokens.append(END_TURN)
                        curr_agent = e.agent
                    if e.action == 'message':
                        # NOTE: both inputs and targets are over entities
                        # TODO: targets over surface tokens
                        tokens.extend((x if isinstance(x, basestring) else x[1] for x in e.processed_data))
                    elif e.action == 'select':
                        tokens.append(SELECT)
                        tokens.append((e.data['Name'].lower(), 'person'))
                    else:
                        raise ValueError('Unknown action.')
                    tokens.append(END_UTTERANCE)
                tokens.append(END_TURN)
                #print 'TOKENS:', tokens
                tokens = map(self.vocab.to_ind, tokens)
                n = len(tokens) - 1
                inputs = np.asarray(tokens[:-1], dtype=dtype).reshape(1, n)
                targets = np.asarray(tokens[1:], dtype=dtype).reshape(1, n)
                #print 'INPUTS:', inputs
                #print 'TARGETS:', targets
                # targets needs to have the same number of dimensions as the output
                targets = np.expand_dims(targets, -1)
                yield inputs, targets

