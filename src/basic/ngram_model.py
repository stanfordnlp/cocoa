__author__ = 'anushabala'
from collections import defaultdict
import numpy as np
from sample_utils import normalize_candidates, sorted_candidates
from ngram_util import dialog_to_message_sequence


class ConditionalProbabilityTable(object):
    '''
    A conditional probability table is a mapping from tuple of strings/tuples to a string/tuple to probability (double).
    '''
    def __init__(self):
        self.data = defaultdict(lambda : defaultdict(float))

    def __getitem__(self, k1):
        return self.data[k1]

    def sample(self, k1):
        tokens = self.data[k1].keys()
        # print k1, tokens
        probs = [self.data[k1][k2] for k2 in tokens]
        idx = np.random.choice(xrange(len(tokens)), p=probs)

        return tokens[idx]

    def normalize(self):
        for k1, m in self.data.items():
            for k2, v in normalize_candidates(m.items()):
                m[k2] = v

    def dump(self):
        for k1, m in self.data.items():
            for k2, v in sorted_candidates(m.items()):
                print '%d\t%s\t%s\t%s' % (len(k1), k1, k2, v)


class NgramModel(object):
    def __init__(self, tagged_data, n=7):
        self.n = n
        self.cpt = ConditionalProbabilityTable()
        self.learn_ngram_model(tagged_data)

    @staticmethod
    def preprocess_tagged_tokens(tagged_utterance):
        """
        Removes the surface form and canonical entity for every tagged token in the utterance. Thus, every tagged token
        is a tuple of (entity_type, feature_tuples) in the list returned by this function. Also replaces the list of
        features by a tuple.
        :param tagged_utterance: A list of tokens/tuples, representing the entities tagged in a single utterance
        (see tagger.py)
        """
        new_utterance = []
        for token in tagged_utterance:
            if not isinstance(token, tuple):
                new_utterance.append(token)
            else:
                # print "In preprocess_tagged_tokens:", token
                _, (_, entity_type, features) = token
                new_utterance.append((entity_type, tuple(features)))

        # print "Original tagged utterance:", tagged_utterance
        # print "Preprocessed utterance:", new_utterance
        return new_utterance

    def preprocess_tagged_dialog(self, tagged_dialog):
        """
        Preprocesses all utterances in a single dialog.
        :param tagged_dialog:
        :return:
        """
        new_dialog = []
        for agent, utterance in tagged_dialog:
            new_dialog.append((agent, self.preprocess_tagged_tokens(utterance)))
        return new_dialog

    def learn_ngram_model(self, tagged_data):
        for dialog in tagged_data:
            processed_dialog = self.preprocess_tagged_dialog(dialog)
            msg_sequence = dialog_to_message_sequence(processed_dialog)

            for i in range(len(msg_sequence)):
                for j in range(max(0, i - self.n), i+1):
                    self.cpt[tuple(msg_sequence[j:i])][msg_sequence[i]] += 1

        self.cpt.normalize()
        # self.cpt.dump()

    def generate(self, history, preprocess=False):
        """
        Generates a token (with features) given a history (list of tagged tokens previously seen). If preprocess=True,
        this function assumes that a tagged token is of the form
        (surface_form, (canonical_entity, entity_type, feature_list)), and preprocesses it to remove the surface form
        and the canonical entity. If not, then the function assumes that every tagged token has the form
        (entity_type, feature_tuple) (where feature_tuple is a tuple of tuples, and every tuple corresponds to a
        single feature - see NgramModel.preprocess_tagged_tokens().)
        :param history: A list of tagged tokens given which to generate a new token
        :param preprocess: Whether to remove entities and surface forms from the history or not
        :return:
        """
        if preprocess:
            # preprocess to remove surface form and canonical entity
            history = self.preprocess_tagged_tokens(history)
        max_tokens = len(history)
        key = tuple()
        for i in np.arange(min(self.n, len(history)), 0, -1):
            key = tuple(history[max_tokens-i:max_tokens])
            if self.cpt[key] is not None and len(self.cpt[key]) > 0:
                break
        return self.cpt.sample(key)

