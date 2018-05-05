'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import time
import os
import numpy as np
import pdb
from itertools import izip, izip_longest
from collections import namedtuple, defaultdict

from cocoa.core.util import read_pickle, write_pickle, read_json
from cocoa.core.entity import Entity, CanonicalEntity, is_entity
from cocoa.lib.bleu import compute_bleu
from cocoa.model.vocab import Vocabulary
from cocoa.model.trie import Trie

from core.price_tracker import PriceTracker, PriceScaler
from core.tokenizer import tokenize
from batcher import DialogueBatcherFactory, Batch, LMBatch
from symbols import markers#, SpecialSymbols
from vocab_builder import create_mappings
from neural import make_model_mappings

def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'type'], default='canonical', help='Output entity form to the decoder')
    parser.add_argument('--candidates-path', nargs='*', default=[], help='Path to json file containing retrieved candidates for dialogues')
    parser.add_argument('--slot-filling', action='store_true', help='Where to do slot filling')
    parser.add_argument('--cache', default='.cache', help='Path to cache for preprocessed batches')
    parser.add_argument('--ignore-cache', action='store_true', help='Ignore existing cache')
    parser.add_argument('--mappings', help='Path to vocab mappings')

START_PRICE = -1
category_to_marker = {
        'car': markers.C_car,
        'phone': markers.C_phone,
        'housing': markers.C_housing,
        'bike': markers.C_bike,
        'furniture': markers.C_furniture,
        'electronics': markers.C_electronics,
        }

def price_filler(x):
    return x == '<price>'


class TextIntMap(object):
    '''
    Map between text and int for visualizing results.
    '''
    def __init__(self, vocab, preprocessor):
        self.vocab = vocab
        self.entity_forms = preprocessor.entity_forms
        self.preprocessor = preprocessor

    def pred_to_input(self, preds, prices=None):
        '''
        Convert decoder outputs to decoder inputs.
        '''
        if self.entity_forms['target'] == self.entity_forms['decoding']:
            return preds
        preds_utterances = [self.int_to_text(pred) for pred in preds]
        # TODO: fill in <price>!!
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

    def int_to_text(self, inds, stage=None, prices=None):
        '''
        Inverse of text_to_int.
        '''
        toks = [self.vocab.to_word(ind) for ind in inds]
        if prices is not None:
            assert len(inds) == len(prices)
            toks = [CanonicalEntity(value=p, type='price') if price_filler(x) else x for x, p in izip(toks, prices)]
        return toks

class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target

    def __init__(self, agent, kb, uuid, model='seq2seq'):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        self.model = model
        self.agent_to_role = self.get_role_mapping(agent, kb)
        # KB context
        # TODO: context_to_int will change category, title, description to integers
        self.category_str = kb.category
        self.category = kb.category
        self.title = tokenize(re.sub(r'[^\w0-9]', ' ', kb.facts['item']['Title']))
        self.description = tokenize(re.sub(r'[^\w0-9]', ' ', ' '.join(kb.facts['item']['Description'])))
        # token_turns: tokens and entitys (output of entity linking)
        self.token_turns = []
        # parsed logical forms
        self.lfs = []
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = [[], [], []]
        # entities: has the same structure as turns, non-entity tokens are None
        self.entities = []
        self.agents = []
        self.roles = []
        self.is_int = False  # Whether we've converted it to integers
        self.num_context = None

        self.price_turns = None

        self.token_candidates = None
        self.candidates = None
        self.true_candidate_inds = None

    @property
    def num_turns(self):
        return len(self.turns[0])

    def join_turns(self):
        for i, utterances in enumerate(self.turns):
            self.turns[i] = [x for utterance in utterances for x in utterance]

    @staticmethod
    def get_role_mapping(agent, kb):
        my_id = agent
        my_role = kb.role

        partner_id = 1 - agent
        partner_role = 'buyer' if my_role == 'seller' else 'seller'

        return {my_id: my_role, partner_id: partner_role}

    def num_tokens(self):
        return sum([len(t) for t in self.token_turns])

    def add_ground_truth(self, candidates, token_turns):
        true_candidate_inds = []
        for cands, turn in izip(candidates, token_turns):
            if not cands:
                inds = []
            else:
                inds = []
                for i, cand in enumerate(cands):
                    if cand == turn:
                        inds.append(i)
                if len(inds) == 0:
                    cands.insert(0, turn)
                    del cands[-1]
                    inds.append(0)
            true_candidate_inds.append(inds)
        return candidates, true_candidate_inds

    def add_candidates(self, candidates, add_ground_truth=False):
        assert len(candidates) == len(self.token_turns)
        if add_ground_truth:
            candidates, self.true_candidate_inds = self.add_ground_truth(candidates, self.token_turns)
        self.token_candidates = candidates

    def add_utterance(self, agent, utterance, lf=None):
        # Always start from the partner agent
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
        self._add_utterance(agent, utterance, lf=lf)

    @classmethod
    def scale_price(cls, kb, utterance):
        return [PriceScaler.scale_price(kb, x) if is_entity(x) else x for x in utterance]

    @classmethod
    def original_price(cls, kb, utterance):
        s = [PriceScaler.unscale_price(kb, x) if is_entity(x) else x for x in utterance]
        return s

    def lf_to_tokens(self, kb, lf):
        intent = lf['intent']
        if intent == 'accept':
            intent = markers.ACCEPT
        elif intent == 'reject':
            intent = markers.REJECT
        elif intent == 'quit':
            intent = markers.QUIT
        elif intent == 'offer':
            intent = markers.OFFER
        tokens = [intent]
        if lf.get('price') is not None:
            p = lf['price']
            price = Entity.from_elements(surface=p, value=p, type='price')
            tokens.append(PriceScaler.scale_price(kb, price).canonical)
        return tokens

    def _insert_markers(self, agent, utterance, new_turn):
        # Mark end of sentence
        utterance.append(markers.EOS)

        # Insert GO
        if new_turn:
            cat_symbol = category_to_marker[self.category_str]
            utterance.insert(0, cat_symbol)

            role = self.agent_to_role[agent]
            start_symbol = markers.GO_S if role == 'seller' else markers.GO_B
            utterance.insert(0, start_symbol)

        return utterance

    def _add_utterance(self, agent, utterance, lf=None):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            new_turn = False
        else:
            new_turn = True

        utterance = self._insert_markers(agent, utterance, new_turn)
        entities = [x if is_entity(x) else None for x in utterance]
        if lf:
            lf = self._insert_markers(agent, self.lf_to_tokens(self.kb, lf), new_turn)
        else:
            lf = []

        if new_turn:
            self.agents.append(agent)
            role = self.agent_to_role[agent]
            self.roles.append(role)

            self.token_turns.append(utterance)
            self.entities.append(entities)
            self.lfs.append(lf)
        else:
            self.token_turns[-1].extend(utterance)
            self.entities[-1].extend(entities)
            self.lfs[-1].extend(lf)

    def candidates_to_int(self):
        def remove_slot(tokens):
            return [x.surface if is_entity(x) and x.canonical.type == 'slot' else x for x in tokens]

        self.candidates = []
        for turn_candidates in self.token_candidates:
            if turn_candidates is None:
                # Encoding turn
                self.candidates.append(None)
            else:
                c = [self.textint_map.text_to_int(remove_slot(c), 'decoding') for c in turn_candidates]
                self.candidates.append(c)

    def kb_context_to_int(self):
        self.category = self.mappings['cat_vocab'].to_ind(self.category)
        self.title = map(self.mappings['kb_vocab'].to_ind, self.title)
        self.description = map(self.mappings['kb_vocab'].to_ind, self.description)

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

        if self.token_candidates:
            self.candidates_to_int()

        #self.price_turns = self.get_price_turns(int_markers.PAD)
        self.kb_context_to_int()
        self.lf_to_int()

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
        self.roles = self._pad_list(self.roles, num_turns, None)
        for turns in self.turns:
            self._pad_list(turns, num_turns, [])
        if self.price_turns:
            self.price_turns = self._pad_list(self.price_turns, num_turns, [])
            assert len(self.price_turns) == len(self.turns[0])
        self.lfs = self._pad_list(self.lfs, num_turns, [])
        if self.candidates:
            self.candidates = self._pad_list(self.candidates, num_turns, [])

    def get_price_turns(self, pad):
        '''
        Given flattened entity turns, return the price for each token.
        pad: used to fill in non-price targets.
        '''
        def to_float_price(entity):
            return float('{:.2f}'.format(PriceTracker.get_price(entity)))
        prices = [[to_float_price(entity) if entity else pad for entity in entities] for entities in self.entities]
        return prices

class Preprocessor(object):
    '''
    Preprocess raw utterances: tokenize, entity linking.
    Convert an Example into a Dialogue data structure used by DataGenerator.
    '''
    def __init__(self, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form, slot_filling=False, slot_detector=None, model='seq2seq'):
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon
        self.entity_forms = {'encoding': entity_encoding_form,
                'decoding': entity_decoding_form,
                'target': entity_target_form}

        if slot_filling:
            assert slot_detector is not None
        self.slot_filling = slot_filling
        self.slot_detector = slot_detector
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
        '''
        Input: utterance is a list of tokens, stage is either encoding, decoding or target
        Output: in most cases, stage will be declared. Based on a combination of
             the model_type and stage, we choose whether or not to summarize the
             utterance.  Models with "sum" should be summarized to only include
             selected keywords, models with "seq" will keep the full sequence.
        '''
        if stage is None:
            return [self.get_entity_form(x, 'canonical') if is_entity(x) else x for x in utterance]
        else:
            if stage == 'encoding':
                summary = self.summarize(utterance) if self.model in ["sum2sum", "sum2seq"] else utterance
            elif (stage == 'decoding') or (stage == 'target'):
                if self.model == "sum2sum":
                    summary = self.summarize(utterance)
                elif self.model == "sum2seq":
                    summary = self.summarize(utterance)
                    summary.append(markers.END_SUM)
                    summary.extend(utterance)
                else:
                    summary = utterance
            return [self.get_entity_form(x, self.entity_forms[stage]) if is_entity(x) else x for x in summary]

    def lf_to_tokens(self, kb, lf):
        intent = lf['intent']
        if intent == 'accept':
            intent = markers.ACCEPT
        elif intent == 'reject':
            intent = markers.REJECT
        elif intent == 'quit':
            intent = markers.QUIT
        elif intent == 'offer':
            intent = markers.OFFER
        tokens = [intent]
        if lf.get('price') is not None:
            p = lf['price']
            price = Entity.from_elements(surface=p, value=p, type='price')
            tokens.append(PriceScaler.scale_price(kb, price))
        return tokens

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue from each agent's perspective
        Create two Dialogue objects for each example
        '''
        kbs = ex.scenario.kbs
        for agent in (0, 1):
            dialogue = Dialogue(agent, kbs[agent], ex.ex_id, model=self.model)
            for e in ex.events:
                if self.model in ('lf2lf', 'lflm'):
                    lf = e.metadata
                    assert lf is not None
                    utterance = self.lf_to_tokens(dialogue.kb, lf)
                else:
                    utterance = self.process_event(e, dialogue.kb)
                if utterance:
                    dialogue.add_utterance(e.agent, utterance, lf=e.metadata)
            yield dialogue

    @classmethod
    def price_to_entity(cls, price):
        return Entity(price, CanonicalEntity(price, 'price'))

    @classmethod
    def _mark_slots(cls, utterance):
        '''
        Insert START_SLOT and END_SLOT around slot words and convert slot entity back
        to normal tokens.
        NOTE: assuming slot words are not joined (see SlotDetector.detect_slot).
        '''
        new_utterance = []
        in_slot = False
        for token in utterance:
            if is_entity(token) and token.canonical.type == 'slot':
                if not in_slot:
                    new_utterance.append(markers.START_SLOT)
                    in_slot = True
                new_utterance.append(token.surface)
            else:
                if in_slot:
                    new_utterance.append(markers.END_SLOT)
                    in_slot = False
                new_utterance.append(token)
        if in_slot:
            new_utterance.append(markers.END_SLOT)
        return new_utterance

    def process_event(self, e, kb):
        '''
        Tokenize, link entities
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.link_entity(tokenize(e.data), kb=kb, scale=True, price_clip=4.)
            if self.slot_filling:
                entity_tokens = self.slot_detector.detect_slots(entity_tokens, kb=kb, join=False)
                entity_tokens = self._mark_slots(entity_tokens)
            if entity_tokens:
                return entity_tokens
            else:
                return None
        elif e.action == 'offer':
            data = e.data['price']
            if data is None:
                return None
            price = PriceScaler._scale_price(kb, data)
            entity_tokens = [markers.OFFER, self.price_to_entity(price)]
            return entity_tokens
        elif e.action == 'quit':
            entity_tokens = [markers.QUIT]
            return entity_tokens
        elif e.action == 'accept':
            entity_tokens = [markers.ACCEPT]
            return entity_tokens
        elif e.action == 'reject':
            entity_tokens = [markers.REJECT]
            return entity_tokens
        else:
            raise ValueError('Unknown event action.')


    def is_keyword(self, token, key_type):
        summary_keywords = {
            "unigram": ["you", "i", "deal", "agree", "great", "!", "?", "n't",
                "can", "not", "have", "good", "bad", "offer", "low", "lower",
                "high", "higher", "sound", "sounds", "price", "but", "give",
                "hello", "hi", "accept"],
            "bigram": [("would",  "you"), ("great", "condition"), ("ca", "n't"),
                ("interested", "in"), ("willing", "to"), ("how", "about"), ("how", "much"),
                ("i", "'m"), ("but", "i"), ("do", "n't"), ("tell", "me"), ("brand", "new")],
            "trigram": [("have", "a", "deal"), ("would", "you", "be"),
                ("i", "can", "do"), ("i", "ca", "n't"), ("pick", "it", "up")]
        }
        return token in summary_keywords[key_type]


    def summarize(self, utterance):
        summary = []
        # number of words possible to loop through in trigrams
        gram_iter = iter(range(len(utterance) - 2))
        # loop through the utterance once to check ngrams
        for i in gram_iter:
            uni = utterance[i]
            bi = tuple(utterance[i:i+2])
            tri = tuple(utterance[i:i+3])

            if is_entity(uni):
                summary.append(uni)
            elif (uni in markers) and (uni != "<pad>"):
                summary.append(uni)
            elif (self.is_keyword(tri, "trigram")):
                for token in tri:
                    summary.append(token)
                next(gram_iter, None)
                next(gram_iter, None)
            elif (self.is_keyword(bi, "bigram")):
                for token in bi:
                    summary.append(token)
                next(gram_iter, None)
            elif (self.is_keyword(uni, "unigram")):
                summary.append(uni)

        # handle the edge cases of last two tokens
        if len(utterance) > 1:
            if self.is_keyword(utterance[-2], "unigram") or is_entity(utterance[-2]):
                summary.append(utterance[-2])
        if self.is_keyword(utterance[-1], "unigram") or is_entity(utterance[-1]):
            summary.append(utterance[-1])

        return summary

    @classmethod
    def skip_example(cls, example):
        tokens = {0: 0, 1: 0}
        turns = {0: 0, 1: 0}
        for event in example.events:
            if event.action == "message":
                msg_tokens = tokenize(event.data)
                tokens[event.agent] += len(msg_tokens)
                turns[event.agent] += 1
        if tokens[0] < 40 and tokens[1] < 40:
            return True
        if turns[0] < 2 or turns[1] < 2:
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
    # TODO: hack
    trie = None

    def __init__(self, train_examples, dev_examples, test_examples, preprocessor,
            args, schema, mappings_path=None, retriever=None, cache='.cache',
            ignore_cache=False, candidates_path=[], num_context=1, batch_size=1,
            trie_path=None, model='seq2seq', add_ground_truth=True):
        examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.iteritems()}
        self.num_context = num_context
        self.model = model

        # Build retriever given training dialogues
        self.retriever = retriever
        self.slot_filling = preprocessor.slot_filling

        self.cache = cache
        self.ignore_cache = ignore_cache
        if (not os.path.exists(cache)) or ignore_cache:
            if retriever is not None:
                self.cached_candidates = self.retriever.load_candidates(candidates_path)
                print 'Cached candidates for %d dialogues' % len(self.cached_candidates)

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

        self.dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model,
                        slot_filling=self.slot_filling,
                        kb_pad=self.mappings['kb_vocab'].to_ind(markers.PAD),
                        mappings=self.mappings, num_context=num_context)

        self.batches = {k: self.create_batches(k, dialogues, batch_size, args.verbose, add_ground_truth=add_ground_truth) for k, dialogues in self.dialogues.iteritems()}
        self.trie = None

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

    def get_dialogue_batch(self, dialogues, slot_filling):
        return DialogueBatcher(dialogues, slot_filling).create_batch()

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

    def get_all_responses(self, name):
        dialogues = self.dialogues[name]
        responses = {'seller': [], 'buyer': []}
        for dialogue in dialogues:
            for turn, role in izip(dialogue.token_turns, dialogue.roles):
                responses[role].extend(turn)
        return responses

    def rewrite_candidate(self, fillers, candidate):
        rewritten = []
        tokens = candidate
        if not tokens:
            return rewritten
        for i, tok in enumerate(tokens):
            if is_entity(tok) and tok.canonical.type == 'slot':
                for filler in fillers:
                    ss = filler.split()
                    new_tokens = list(tokens)
                    del new_tokens[i]
                    for j, s in enumerate(ss):
                        new_tokens.insert(i+j, Entity(s, CanonicalEntity('', 'slot')))
                    new_cand = new_tokens
                    rewritten.append(new_cand)
        return rewritten

    def create_batches(self, name, dialogues, batch_size, verbose, add_ground_truth=True):
        if not os.path.isdir(self.cache):
            os.makedirs(self.cache)
        cache_file = os.path.join(self.cache, '%s_batches.pkl' % name)
        if (not os.path.exists(cache_file)) or self.ignore_cache:
            if self.retriever is not None:
                for dialogue in dialogues:
                    k = (dialogue.uuid, dialogue.role)
                    # candidates: list of list containing num_candidates responses for each turn of the dialogue
                    # None candidates means that it is not the agent's speaking turn
                    if k in self.cached_candidates:
                        candidates = self.cached_candidates[k]
                    else:
                        candidates = self.retriever.retrieve_candidates(dialogue, json_dict=False)

                    dialogue.add_candidates(candidates, add_ground_truth=add_ground_truth)
                self.retriever.report_search_time()

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
        # TODO: hack
        if self.model == 'lflm':
            for ind in inds:
                for batch in dialogue_batches[ind]:
                    yield LMBatch(batch['inputs'],
                                  batch['targets'],
                                  self.mappings['utterance_vocab'],
                                  cuda=cuda)
        else:
            for ind in inds:
                for batch in dialogue_batches[ind]:
                    yield Batch(batch['encoder_args'],
                                batch['decoder_args'],
                                batch['context_data'],
                                self.mappings['utterance_vocab'],
                                num_context=self.num_context, cuda=cuda)
                # End of dialogue
                yield None

    def create_trie(self, batches, path):
        if path is None:
            return None
        def seq_iter(batches):
            for batch in batches:
                for b in batch['batch_seq']:
                    targets = b['targets']
                    for target in targets:
                        yield target
        if not os.path.exists(path):
            trie = Trie()
            print 'Build prefix trie of length', trie.max_prefix_len
            start_time = time.time()
            trie.build_trie(seq_iter(batches))
            print '[%d s]' % (time.time() - start_time)
            print 'Write trie to', path
            write_pickle(trie, path)
        else:
            print 'Read trie from', path
            trie = read_pickle(path)
        return trie


class LMDataGenerator(DataGenerator):
    def create_dialogue_batches(self, dialogues, batch_size, bptt_steps=35):
        dialogue_batches = []
        # TODO: only need diaogue from one direction
        dialogues.sort(key=lambda d: d.num_tokens())
        N = len(dialogues)
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.append(LMDialogueBatcher(dialogue_batch).create_batch(bptt_steps))
            start = end
        return dialogue_batches


