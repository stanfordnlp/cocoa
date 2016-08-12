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
                curr_message = [START]
                write = []  # write when agent == 0
                for i, e in enumerate(ex.events):
                    if e.action == 'message':
                        # NOTE: both inputs and targets are over entities
                        # TODO: targets over surface tokens
                        curr_message.extend((x if isinstance(x, basestring) else x[1] for x in e.processed_data))
                    elif e.action == 'select':
                        curr_message.append(SELECT)
                        curr_message.append((e.data['Name'].lower(), 'person'))
                    else:
                        raise ValueError('Unknown action.')
                    curr_message.append(END_UTTERANCE)
                    if i+1 == len(ex.events) or e.agent != ex.events[i+1].agent:
                        curr_message.append(END_TURN)
                        write.extend([True if e.agent == 0 else False] * len(curr_message))
                        tokens.extend(curr_message)
                        curr_message = []
                #print 'TOKENS:', tokens
                tokens = map(self.vocab.to_ind, tokens)
                n = len(tokens) - 1
                inputs = np.asarray(tokens[:-1], dtype=dtype).reshape(1, n)
                targets = np.asarray(tokens[1:], dtype=dtype).reshape(1, n)
                #print 'INPUTS:', inputs
                #print 'TARGETS:', targets
                #print 'WRITE:', write
                # write when agent == 0
                yield inputs, targets, np.asarray(write[1:], dtype=np.bool_).reshape(1, n)
                # write when agent == 1
                yield inputs, targets, np.asarray([False if w else True for w in write[1:]], dtype=np.bool_).reshape(1, n)

