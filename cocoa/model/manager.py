from cocoa.core.util import read_pickle, write_pickle
from cocoa.model.counter import build_vocabulary, count_ngrams
from cocoa.model.ngram import MLENgramModel

class Manager(object):
    def __init__(self, model, actions):
        self.model = model
        self.actions = actions

    @classmethod
    def from_train(cls, sequences, n=3):
        vocab = build_vocabulary(1, *sequences)
        counter = count_ngrams(n, vocab, sequences, pad_left=True, pad_right=False)
        model = MLENgramModel(counter)
        actions = vocab.keys()
        #print model.score('init-price', ('<start>',))
        #print model.ngrams.most_common(10)
        return cls(model, actions)

    def available_actions(self, state):
        actions = [a for a in self.actions if a != 'unknown']
        return actions

    def choose_action(self, state, context=None):
        if not context:
            context = (state.my_act, state.partner_act)
        freqdist = self.model.freqdist(context)
        actions = self.available_actions(state)
        freqdist = [x for x in freqdist if x[0] in actions]
        # TODO: backoff
        if len(freqdist) == 0:
            return None
        best_action = max(freqdist, key=lambda x: x[1])[0]
        print 'context:', context
        #print 'dist:', freqdist
        print 'availabel actions:', actions
        print 'action:', best_action
        return best_action

    def save(self, output):
        data = {'model': self.model, 'actions': self.actions}
        write_pickle(data, output)

    @classmethod
    def from_pickle(cls, path):
        data = read_pickle(path)
        return cls(data['model'], data['actions'])
