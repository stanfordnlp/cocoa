"""Functions that analyze dialogues and models.
"""
import json
from collections import defaultdict
import numpy as np

from cocoa.core.entity import is_entity

from core.tokenizer import tokenize

class Analyzer(object):
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def example_stats(self, examples):
        stats = {}
        stats['num_dialogues'] = len(examples)
        stats['num_turns_per_dialogue'] = np.mean([len(e.events) for e in examples])
        utterances = [tokenize(e.data) \
                for example in examples \
                for e in example.events if e.action == 'message']
        stats['num_tokens_per_turn'] = np.mean([len(u) for u in utterances])
        vocab = set()
        for u in utterances:
            vocab.update(u)
        stats['vocab_size'] = len(vocab)
        self.print_stats(stats, 'dataset stats')
        return stats

    def print_stats(self, stats, name):
        print '='*5, name.upper(), '='*5
        print json.dumps(stats, indent=2)

    def parser_stats(self, parsed_dialogues):
        stats = {}
        non_entity_vocab = set()
        stats['intents'] = defaultdict(int)
        for dialogue in parsed_dialogues:
            for utterance in dialogue:
                if utterance.tokens is not None:
                    tokens = [x.canonical.type if is_entity(x) else x for x in utterance.tokens]
                    non_entity_vocab.update(tokens)
                if utterance.lf and utterance.lf.intent != '<start>':
                    stats['intents'][utterance.lf.intent] += 1
        stats['non_entity_vocab_size'] = len(non_entity_vocab)

        # Percentage intents
        s = float(sum(stats['intents'].values()))
        stats['intents'] = sorted(
                [(k, v, v / s) for k, v in stats['intents'].iteritems()],
                key=lambda x: x[1], reverse=True)

        self.print_stats(stats, 'parser stats')
        return stats

    def entropy(self, p, normalized=True):
        p = np.array(p, dtype=np.float32)
        if not normalized:
            p /= np.sum(p)
        ent = -1. * np.sum(p * np.log(p))
        return ent

    def manager_stats(self, manager):
        stats = {}
        stats['actions'] = manager.actions

        # Most likely sequence
        action_seq = ['<start>', '<start>']
        entropy_seq = []
        for i in xrange(10):
            context = tuple(action_seq[-2:])

            freqdist = manager.model.freqdist(context)
            counts = [x[1] for x in freqdist]
            ent = self.entropy(counts, normalized=False)
            entropy_seq.append((context, ent))

            action = manager.choose_action_from_freqdist(context)
            action_seq.append(action)

        stats['action_seq'] = action_seq
        stats['entropy_seq'] = entropy_seq

        self.print_stats(stats, 'manager stats')
        return stats
