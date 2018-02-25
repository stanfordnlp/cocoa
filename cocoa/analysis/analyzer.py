"""Functions that analyze dialogues and models.
"""
import json
from collections import defaultdict
import numpy as np

from cocoa.core.entity import is_entity
from cocoa.model.util import entropy, safe_div
from cocoa.model.counter import build_vocabulary, count_ngrams
from cocoa.model.ngram import MLENgramModel

from core.tokenizer import tokenize

all_vocab = None
no_ent_vocab = None

class Analyzer(object):
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def example_stats(self, examples, agent=None):
        stats = {}
        stats['num_dialogues'] = len(examples)
        stats['num_turns_per_dialogue'] = np.mean([len(e.events) for e in examples])
        utterances = [tokenize(e.data) \
                for example in examples \
                    for e in example.events if e.action == 'message' and
                        (not agent or example.agents[e.agent] == agent)]
        stats['num_tokens_per_turn'] = np.mean([len(u) for u in utterances])

        vocab = set()
        for u in utterances:
            vocab.update(u)
        stats['vocab_size'] = len(vocab)
        global all_vocab
        all_vocab = vocab
        stats['corpus_perplexity'] = self.sequence_perplexity(utterances)

        self.print_stats(stats, 'dataset stats')
        return stats

    def intent_sequence_perplexity(self, intent_sequences, n=3):
        H = 0.
        N = 0
        for intent, sequences in intent_sequences.iteritems():
            model = self.build_lm(sequences, n)
            H_, N_ = self.total_entropy(model, sequences)
            H += H_
            N += N_
        H = safe_div(H, N)
        return np.power(2, H)

    def total_entropy(self, model, sequences):
        H = 0.
        N = 0
        for s in sequences:
            h, n = model.entropy(s, average=False)
            H += h
            N += n
        return H, N

    def build_lm(self, sequences, n):
        vocab = build_vocabulary(1, *sequences)
        counter = count_ngrams(n, vocab, sequences, pad_left=True, pad_right=False)
        model = MLENgramModel(counter)
        return model

    def sequence_perplexity(self, sequences, n=3):
        model = self.build_lm(sequences, n)
        H, N = self.total_entropy(model, sequences)
        H = safe_div(H, N)
        return np.power(2, H)

    def print_stats(self, stats, name):
        print '='*5, name.upper(), '='*5
        print json.dumps(stats, indent=2)

    def parser_stats(self, parsed_dialogues, agent=None):
        stats = {}
        non_entity_vocab = set()
        ents = set()
        stats['intents'] = defaultdict(int)
        intent_utterances = defaultdict(list)

        for dialogue in parsed_dialogues:
            for utterance in dialogue:
                if agent and utterance.agent != agent:
                    continue
                if utterance.tokens is not None:
                    tokens = [x.canonical.type if is_entity(x) else x for x in utterance.tokens]
                    e = [x.surface for x in utterance.tokens if is_entity(x)]
                    ents.update(e)
                    non_entity_vocab.update(tokens)
                if utterance.lf and utterance.lf.intent != '<start>':
                    stats['intents'][utterance.lf.intent] += 1
                if utterance.text is not None:
                    intent_utterances[utterance.lf.intent].append(tokenize(utterance.text))
        stats['non_entity_vocab_size'] = len(non_entity_vocab)
        #print 'entities:', len(ents)
        #global no_ent_vocab
        #no_ent_vocab = non_entity_vocab
        #for x in all_vocab:
        #    if not x in non_entity_vocab:
        #        print x

        stats['intent_corpus_perplexity'] = self.intent_sequence_perplexity(intent_utterances)

        # Percentage intents
        #s = float(sum(stats['intents'].values()))
        #stats['intents'] = sorted(
        #        [(k, v, v / s) for k, v in stats['intents'].iteritems()],
        #        key=lambda x: x[1], reverse=True)

        self.print_stats(stats, 'parser stats')
        return stats

    def manager_stats(self, manager):
        stats = {}
        stats['actions'] = manager.actions

        # Most likely sequence
        action_seq = [{'context': ('<start>', '<start>')}]
        for i in xrange(10):
            state = action_seq[-1]
            context = state['context']

            freqdist = manager.model.freqdist(context)
            counts = [x[1] for x in freqdist]
            ent = entropy(counts, normalized=False)
            state['entropy'] = ent

            state['most_likely_action'] = manager.most_likely_action(context, freqdist)
            state['min_entropy_action'] = manager.min_entropy_action(context, freqdist)

            new_context = (context[-1], state['most_likely_action'])
            action_seq.append({'context': new_context})

        stats['action_seq'] = action_seq

        self.print_stats(stats, 'manager stats')
        return stats

    #def generator_stats(self, generator):
