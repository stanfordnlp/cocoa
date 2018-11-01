'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import time
import os
import numpy as np
from itertools import izip, izip_longest
from collections import namedtuple, defaultdict
import copy

from cocoa.core.util import read_pickle, write_pickle, read_json
from cocoa.core.entity import Entity, CanonicalEntity, is_entity
from cocoa.lib.bleu import compute_bleu
from cocoa.model.vocab import Vocabulary

from core.tokenizer import tokenize
from batcher import DialogueBatcherFactory, Batch
from symbols import markers
from vocab_builder import create_mappings
from neural import make_model_mappings

class TextIntMap(object):
    '''
    Map between text and int for visualizing results.
    '''
    def __init__(self, vocab, preprocessor):
        self.vocab = vocab
        self.entity_forms = preprocessor.entity_forms
        self.preprocessor = preprocessor

    def pred_to_input(self, preds):
        '''
        Convert decoder outputs to decoder inputs.
        '''
        if self.entity_forms['target'] == self.entity_forms['decoding']:
            return preds
        preds_utterances = [self.int_to_text(pred) for pred in preds]
        input_utterances = [self.preprocessor.process_utterance(utterance, 'decoding') for utterance in preds_utterances]
        inputs = np.array([self.text_to_int(utterance, 'decoding') for utterance in input_utterances])
        return inputs

    def text_to_int(self, utterance, stage=None):
        '''
        Process entities in the utterance based on whether it is used for encoding, decoding
        or ground truth.
        '''
        tokens = self.preprocessor.process_utterance(utterance, stage)
        return [self.vocab.to_ind(token) for token in tokens]

    def int_to_text(self, inds, stage=None):
        '''
        Inverse of text_to_int.
        '''
        toks = [self.vocab.to_word(ind) for ind in inds]
        return toks

class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target

    def __init__(self, agent, kb, outcome, uuid, model='seq2seq'):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        self.model = model
        self.scenario = self.process_scenario(kb)
        self.selection = self.process_selection(outcome['item_split'])
        # token_turns: tokens and entitys (output of entity linking)
        self.token_turns = []
        # parsed logical forms
        self.lfs = []
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = [[], [], []]
        # entities: has the same structure as turns, non-entity tokens are None
        self.entities = []
        self.agents = []
        self.is_int = False  # Whether we've converted it to integers
        self.num_context = None

    @property
    def num_turns(self):
        return len(self.turns[0])

    def join_turns(self):
        for i, utterances in enumerate(self.turns):
            self.turns[i] = [x for utterance in utterances for x in utterance]

    def num_tokens(self):
        return sum([len(t) for t in self.token_turns])

    def add_utterance(self, agent, utterance, lf=None):
        # Always start from the partner agent
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
        self._add_utterance(agent, utterance, lf=lf)

    def _add_utterance(self, agent, utterance, lf=None):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            new_turn = False
        else:
            new_turn = True

        utterance = self._insert_markers(agent, utterance, new_turn)
        entities = [x if is_entity(x) else None for x in utterance]
        if lf:
            lf = self._insert_markers(agent, self.lf_to_tokens(lf), new_turn)
        else:
            lf = []

        if new_turn:
            self.agents.append(agent)

            self.token_turns.append(utterance)
            self.entities.append(entities)
            self.lfs.append(lf)
        else:
            self.token_turns[-1].extend(utterance)
            self.entities[-1].extend(entities)
            self.lfs[-1].extend(lf)

    def lf_to_tokens(self, lf, proposal=None, items=None):
        intent = lf['intent']
        if intent == 'select':
            intent = markers.SELECT
        elif intent == 'quit':
            intent = markers.QUIT
        tokens = [intent]
        if proposal is not None:
            for item in items:
                tokens.append('{item}={count}'.format(item=item, count= proposal['me'][item]))
                #tokens.append('{count}'.format(count= proposal['me'][item]))
                #tokens.append(str(proposal[item]))
        return tokens

    def _insert_markers(self, agent, utterance, new_turn):
        ''' Add start of sentence and end of sentence markers, ignore other
        markers which were specific to craigslist'''
        utterance.append(markers.EOS)

        if new_turn:
            utterance.insert(0, markers.GO)

        return utterance

    def scenario_to_int(self):
        self.scenario = map(self.mappings['kb_vocab'].to_ind, self.scenario)

    def process_scenario(self, kb):
        attributes = ("Count", "Value")  # "Name"
        scenario = ['{item}-{attr}-{value}'.format(item=fact['Name'], attr=attr, value=fact[attr])
            for fact in kb.items for attr in attributes]
        assert(len(scenario) == 6)
        return scenario

    def selection_to_int(self):
        # TODO: have a different vocab
        self.selection = map(self.mappings['utterance_vocab'].to_ind, self.selection)

    def process_selection(self, item_split):
        selection = []
        for agent, agent_name in izip((self.agent, 1-self.agent), ('my', 'your')):
            for item in ("book", "hat", "ball"):
                selection.append('{item}={count}'.format(
                    item=item, count=item_split[agent][item]))
        assert(len(selection) == 6)
        return selection

    def lf_to_int(self):
        self.lf_token_turns = []
        for i, lf in enumerate(self.lfs):
            self.lf_token_turns.append(lf)
            self.lfs[i] = map(self.mappings['lf_vocab'].to_ind, lf)

    def convert_to_int(self):
        if self.is_int:
            return

        for turn in self.token_turns:
            # turn is a list of tokens that an agent spoke on their turn
            # self.turns starts out as [[], [], []], so
            #   each portion is a list holding the tokens of either the
            #   encoding portion, decoding portion, or the target portion
            for portion, stage in izip(self.turns, ('encoding', 'decoding', 'target')):
                portion.append(self.textint_map.text_to_int(turn, stage))

        self.scenario_to_int()
        self.selection_to_int()

        self.is_int = True

    def _pad_list(self, l, size, pad):
        for i in xrange(len(l), size):
            l.append(pad)
        return l

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        self.agents = self._pad_list(self.agents, num_turns, None)
        for turns in self.turns:
            self._pad_list(turns, num_turns, [])
        self.lfs = self._pad_list(self.lfs, num_turns, [])

class Preprocessor(object):
    '''
    Preprocess raw utterances: tokenize, entity linking.
    Convert an Example into a Dialogue data structure used by DataGenerator.
    '''
    def __init__(self, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form, model='seq2seq'):
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon
        self.entity_forms = {'encoding': entity_encoding_form,
                'decoding': entity_decoding_form,
                'target': entity_target_form}
        self.model = model

    @classmethod
    def get_entity_form(cls, entity, form):
        '''
        An entity is represented as (surface_form, (canonical_form, type)).
        '''
        assert len(entity) == 2
        if form == 'surface':
            return entity.surface
        elif form == 'type':
            return '<%s>' % entity.canonical.type
        elif form == 'canonical':
            return entity._replace(surface='')
        else:
            raise ValueError('Unknown entity form %s' % form)

    def process_utterance(self, utterance, stage=None):
        # Input: utterance is a list of tokens, stage is either encoding, decoding or target
        if stage is None:
            return [self.get_entity_form(x, 'canonical') if is_entity(x) else x for x in utterance]
        else:
            return [self.get_entity_form(x, self.entity_forms[stage]) if is_entity(x) else x for x in utterance]

    def process_event(self, e, agent, sel=None):
        # Lower, tokenize, link entity
        if e.action == 'message':
            entity_tokens = self.lexicon.link_entity(tokenize(e.data))
            return entity_tokens if entity_tokens else None
        elif e.action == 'select':
            '''
            outcome is now handled separately with its own loss function
            ---- ABOVE IS OUTDATED ----
            "sel" is short for selection, it will be a list of two agents
            where each agent is a dict with 3 keys: book, hat, ball
            '''
            entity_tokens = [markers.SELECT]
            if e.agent == agent:
                entity_tokens.extend(['{count}'.format(count=sel[agent][item]) for item in self.lexicon.items])
            ## First my selection, then partner selection
            #selections = []
            #for agent in (e.agent, 1 - e.agent):
            #    selections.extend([sel[agent][item] for item in self.lexicon.items])
            #entity_tokens.extend([str(x) for x in selections])
            return entity_tokens
        elif e.action == 'quit':
            entity_tokens = [markers.QUIT]
            return entity_tokens
        else:
            raise ValueError('Unknown event action.')

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue from each agent's perspective
        Create two Dialogue objects for each example
        '''
        for agent in (0, 1):
            dialogue = Dialogue(agent, ex.scenario.kbs[agent],
                            ex.outcome, ex.ex_id, model=self.model)
            for e in ex.events:
                if self.model in ('lf2lf', 'lflm'):
                    lf = e.metadata
                    proposal = None
                    assert lf is not None
                    # TODO: hack
                    if lf.get('proposal') is not None:
                        proposal = {'me': {}, 'you': {}}
                        # Parser is from the receiver's perspective
                        received_proposal = {int(k): v for k, v in lf['proposal'].iteritems()}
                        proposal['me'] = received_proposal[dialogue.agent]
                        proposal['you'] = received_proposal[1-dialogue.agent]
                    if e.action == 'select':
                        if e.agent != dialogue.agent:
                            proposal = None
                        else:
                            sel = ex.outcome['item_split']
                            proposal = {'me': {}, 'you': {}}
                            for item, count in sel[dialogue.agent].iteritems():
                                proposal['me'][item] = count
                            for item, count in sel[1-dialogue.agent].iteritems():
                                proposal['you'][item] = count
                    #if proposal:
                    #    if e.agent == dialogue.agent:
                    #        proposal = proposal['me']
                    #    else:
                    #        proposal = proposal['you']
                    utterance = dialogue.lf_to_tokens(lf, proposal, items=self.lexicon.items)
                else:
                    sel = ex.outcome['item_split']
                    utterance = self.process_event(e, dialogue.agent, sel)
                if utterance:
                    dialogue.add_utterance(e.agent, utterance, lf=e.metadata)
            yield dialogue

    @classmethod
    # insure we have full dialogues with minimum number of tokens
    def skip_example(cls, example):
        if not example.outcome['valid_deal'] :
            return True
        return False

    def preprocess(self, examples):
        dialogues = []
        for ex in examples:
            if self.skip_example(ex):
                continue
            for d in self._process_example(ex):
                dialogues.append(d)
        return dialogues

class DataGenerator(object):

    def __init__(self, train_examples, dev_examples, test_examples, preprocessor,
            args, schema, mappings_path=None, cache='.cache',
            ignore_cache=False, num_context=1, batch_size=1,
            model='seq2seq'):
        examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.iteritems()}
        self.num_context = num_context
        self.model = model

        self.cache = cache
        self.ignore_cache = ignore_cache
        if (not os.path.exists(cache)) or ignore_cache:
            # NOTE: each dialogue is made into two examples from each agent's perspective
            self.dialogues = {k: preprocessor.preprocess(v)  for k, v in examples.iteritems() if v}

            for fold, dialogues in self.dialogues.iteritems():
                print '%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold])
        else:
            self.dialogues = {k: None  for k, v in examples.iteritems() if v}
            print 'Using cached data from', cache

        self.mappings = self.load_mappings(model, mappings_path, schema, preprocessor)
        self.textint_map = TextIntMap(self.mappings['utterance_vocab'], preprocessor)

        Dialogue.mappings = self.mappings
        Dialogue.textint_map = self.textint_map
        Dialogue.preprocessor = preprocessor
        Dialogue.num_context = num_context

        #global int_markers
        #int_markers = SpecialSymbols(*[self.mappings['utterance_vocab'].to_ind(m) for m in markers])
        # kb is alwyas 6 numbers, no padding
        self.dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model,
                kb_pad=None,
                mappings=self.mappings, num_context=num_context)

        self.batches = {k: self.create_batches(k, dialogues, batch_size, args.verbose) for k, dialogues in self.dialogues.iteritems()}

    def load_mappings(self, model_type, mappings_path, schema, preprocessor):
        vocab_path = os.path.join(mappings_path, 'vocab.pkl')
        if not os.path.exists(vocab_path):
            print 'Vocab not found at', vocab_path
            mappings = create_mappings(self.dialogues['train'], schema,
                preprocessor.entity_forms.values())
            write_pickle(mappings, vocab_path)
            print('Wrote mappings to {}, now exiting.'.format(vocab_path))
            import sys; sys.exit()
        else:
            print 'Loading vocab from', vocab_path
            mappings = read_pickle(vocab_path)
            for k, v in mappings.iteritems():
                print k, v.size
            mappings = make_model_mappings(model_type, mappings)
            return mappings

    def convert_to_int(self):
        '''
        Convert tokens to integers.
        '''
        for fold, dialogues in self.dialogues.iteritems():
            for dialogue in dialogues:
                dialogue.convert_to_int()

    def get_dialogue_batch(self, dialogues):
        return DialogueBatcher(dialogues).create_batch()

    def dialogue_sort_score(self, d):
        # Sort dialogues by number o turns
        return len(d.turns[0])

    def create_dialogue_batches(self, dialogues, batch_size):
        dialogue_batches = []
        dialogues.sort(key=lambda d: self.dialogue_sort_score(d))
        N = len(dialogues)
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.append(self.dialogue_batcher.create_batch(dialogue_batch))
            start = end
        return dialogue_batches

    def create_batches(self, name, dialogues, batch_size, verbose):
        if not os.path.isdir(self.cache):
            os.makedirs(self.cache)
        cache_file = os.path.join(self.cache, '%s_batches.pkl' % name)
        if (not os.path.exists(cache_file)) or self.ignore_cache:
            random.shuffle(dialogues)
            for dialogue in dialogues:
                dialogue.convert_to_int()

            dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)
            print 'Write %d batches to cache %s' % (len(dialogue_batches), cache_file)
            start_time = time.time()
            write_pickle(dialogue_batches, cache_file)
            print '[%d s]' % (time.time() - start_time)
        else:
            start_time = time.time()
            dialogue_batches = read_pickle(cache_file)
            if verbose:
                print 'Read %d batches from cache %s' % (len(dialogue_batches), cache_file)
                print '[%d s]' % (time.time() - start_time)
        return dialogue_batches

    def generator(self, name, shuffle=True, cuda=True):
        dialogue_batches = self.batches[name]
        yield sum([len(b) for b in dialogue_batches])
        inds = range(len(dialogue_batches))
        if shuffle:
            random.shuffle(inds)
        for ind in inds:
            for batch in dialogue_batches[ind]:
                yield Batch(batch['encoder_args'],
                            batch['decoder_args'],
                            batch['context_data'],
                            self.mappings['utterance_vocab'],
                            num_context=self.num_context, cuda=cuda)
            # End of dialogue
            yield None

