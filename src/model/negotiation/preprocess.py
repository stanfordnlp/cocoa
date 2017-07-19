'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import time
import os
import numpy as np
from src.basic.util import read_pickle, write_pickle, read_json
from src.model.vocab import Vocabulary
from src.basic.entity import Entity, CanonicalEntity, is_entity
from itertools import chain, izip, izip_longest
from collections import namedtuple, defaultdict, deque
import copy
from src.basic.negotiation.price_tracker import PriceTracker, PriceScaler
from src.basic.negotiation.tokenizer import tokenize
from src.lib.bleu import compute_bleu

def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['type', 'canonical'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'type'], default='canonical', help='Output entity form to the decoder')
    parser.add_argument('--candidates-path', nargs='*', default=[], help='Path to json file containing retrieved candidates for dialogues')
    parser.add_argument('--slot-filling', action='store_true', help='Where to do slot filling')
    parser.add_argument('--cache', default='.cache', help='Path to cache for preprocessed batches')
    parser.add_argument('--ignore-cache', action='store_true', help='Ignore existing cache')

SpecialSymbols = namedtuple('SpecialSymbols', ['EOS', 'GO_S', 'GO_B', 'OFFER', 'QUIT', 'ACCEPT', 'REJECT', 'PAD', 'START_SLOT', 'END_SLOT'])
markers = SpecialSymbols(EOS='</s>', GO_S='<go-s>', GO_B='<go-b>', OFFER='<offer>', QUIT='<quit>', ACCEPT='<accept>', REJECT='<reject>', PAD='<pad>', START_SLOT='<slot>', END_SLOT='</slot>')
START_PRICE = -1

def price_filler(x):
    return x == '<price>'

def build_vocab(dialogues, special_symbols=[], entity_forms=[]):
    vocab = Vocabulary(offset=0, unk=True)

    def _add_entity(entity):
        for entity_form in entity_forms:
            word = Preprocessor.get_entity_form(entity, entity_form)
            vocab.add_word(word)

    # Add words
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for turn in dialogue.token_turns:
            #for token in chain.from_iterable(turn):
            for token in turn:
                if is_entity(token):
                    _add_entity(token)
                else:
                    vocab.add_word(token)

    # Add special symbols
    vocab.add_words(special_symbols)
    print 'Utterance vocabulary size:', vocab.size
    return vocab

def build_kb_vocab(dialogues, special_symbols=[]):
    vocab = Vocabulary(offset=0, unk=True)
    cat_vocab = Vocabulary(offset=0, unk=False)
    cat_vocab.add_words(['bike', 'car', 'electronics', 'furniture', 'housing', 'phone'])
    for dialogue in dialogues:
        assert dialogue.is_int is False
        # TODO: remove symbols, puncts
        vocab.add_words(dialogue.title)
        vocab.add_words(dialogue.description)
        cat_vocab.add_word(dialogue.category)
    print 'KB vocabulary size:', vocab.size
    print 'Category size:', cat_vocab.size
    return vocab, cat_vocab

def create_mappings(dialogues, schema, entity_forms):
    vocab = build_vocab(dialogues, markers, entity_forms)
    kb_vocab, cat_vocab = build_kb_vocab(dialogues, [markers.PAD])
    return {'vocab': vocab,
            'kb_vocab': kb_vocab,
            'cat_vocab': cat_vocab,
            }

def pad_list_to_array(l, fillvalue, dtype):
    '''
    l: list of lists with unequal length
    return: np array with minimal padding
    '''
    return np.array(list(izip_longest(*l, fillvalue=fillvalue)), dtype=dtype).T

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
    num_candidates = None

    def __init__(self, agent, kb, uuid):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        self.role = kb.facts['personal']['Role']
        partner_role = 'buyer' if self.role == 'seller' else 'seller'
        self.agent_to_role = {self.agent: self.role, 1 - self.agent: partner_role}
        # KB context
        self.category = kb.facts['item']['Category']
        self.title = tokenize(re.sub(r'[^\w0-9]', ' ', kb.facts['item']['Title']))
        self.description = tokenize(re.sub(r'[^\w0-9]', ' ', ' '.join(kb.facts['item']['Description'])))
        # token_turns: tokens and entitys (output of entitylink)
        self.token_turns = []
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = [[], [], []]
        # entities: has the same structure as turns, non-entity tokens are None
        self.entities = []
        self.agents = []
        self.roles = []
        self.is_int = False  # Whether we've converted it to integers
        self.token_candidates = None
        self.candidates = None

    def join_turns(self):
        all_turns = []
        role_to_id = {'buyer': int_markers.GO_B, 'seller': int_markers.GO_S}
        for agent, turn in izip(self.agents, self.turns[0]):
            start_symbol = role_to_id[self.agent_to_role[agent]]
            all_turns.append([start_symbol] + turn)
        all_tokens = [x for turn in all_turns for x in turn]
        return all_tokens, all_turns

    def num_tokens(self):
        return sum([len(t) for t in self.token_turns])

    def add_candidates(self, candidates):
        assert len(candidates) == len(self.token_turns)
        self.token_candidates = candidates

    def add_utterance(self, agent, utterances):
        # Always start from the partner agent
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [])
        self._add_utterance(agent, utterances)

    @classmethod
    def scale_price(cls, kb, utterance):
        return [PriceScaler.scale_price(kb, x) if is_entity(x) else x for x in utterance]

    @classmethod
    def original_price(cls, kb, utterance):
        s = [cls._original_price(kb, x) if is_entity(x) else x for x in utterance]
        return s

    def _insert_markers(self, agent, utterance, new_turn):
        # Mark end of sentence
        utterance.append(markers.EOS)

        # Insert GO
        if new_turn:
            role = self.agent_to_role[agent]
            start_symbol = markers.GO_S if role == 'seller' else markers.GO_B
            utterance.insert(0, start_symbol)

        return utterance

    def _add_utterance(self, agent, utterance):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            new_turn = False
        else:
            new_turn = True

        utterance = self.scale_price(self.kb, utterance)
        utterance = self._insert_markers(agent, utterance, new_turn)
        entities = [x if is_entity(x) else None for x in utterance]

        if not new_turn:
            self.token_turns[-1].extend(utterance)
            self.entities[-1].extend(entities)
        else:
            self.agents.append(agent)
            role = self.agent_to_role[agent]
            self.roles.append(role)

            self.token_turns.append(utterance)
            self.entities.append(entities)

    def convert_to_int(self):
        if self.is_int:
            return

        for turn in self.token_turns:
            for turns, stage in izip(self.turns, ('encoding', 'decoding', 'target')):
                #turns.append([self.textint_map.text_to_int(utterance, stage)
                #    for utterance in turn])
                turns.append(self.textint_map.text_to_int(turn, stage))

        def remove_slot(tokens):
            return [x.surface if is_entity(x) and x.canonical.type == 'slot' else x for x in tokens]

        if self.token_candidates:
            self.candidates = []
            for turn_candidates in self.token_candidates:
                c = [self.textint_map.text_to_int(remove_slot(c['response']), 'decoding') if 'response' in c else [] for c in turn_candidates]
                self.candidates.append(c)

        self.price_turns = self.get_price_turns(int_markers.PAD)
        self.category = self.mappings['cat_vocab'].to_ind(self.category)
        self.title = map(self.mappings['kb_vocab'].to_ind, self.title)
        self.description = map(self.mappings['kb_vocab'].to_ind, self.description)
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
        self.price_turns = self._pad_list(self.price_turns, num_turns, [])
        if self.candidates:
            self.candidates = self._pad_list(self.candidates, num_turns, [{} for _ in xrange(self.num_candidates)])
        assert len(self.price_turns) == len(self.turns[0])

    def get_price_turns(self, pad):
        '''
        Given flattened entity turns, return the price for each token.
        pad: used to fill in non-price targets.
        '''
        def to_float_price(entity):
            return float('{:.2f}'.format(PriceTracker.get_price(entity)))
        prices = [[to_float_price(entity) if entity else pad for entity in entities] for entities in self.entities]
        return prices

class DialogueBatch(object):
    def __init__(self, dialogues, slot_filling):
        self.dialogues = dialogues
        self.slot_filling = slot_filling

    def _normalize_dialogue(self):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([len(d.turns[0]) for d in self.dialogues])
        for dialogue in self.dialogues:
            dialogue.pad_turns(max_num_turns)
        self.num_turns = len(self.dialogues[0].turns[0])

    @classmethod
    def _normalize_candidates(cls, candidate_batch, roles):
        '''
        All turns at the same time step should have the same number of tokens.
        return: (batch_size, num_candidates, candidate_len)
        '''
        num_candidates = len(candidate_batch[0])  # Same for all instance
        max_num_tokens = max([max([len(c) for c in candidates]) for candidates in candidate_batch])
        batch_size = len(candidate_batch)
        T = np.full([batch_size, num_candidates, max_num_tokens+1], int_markers.PAD, dtype=np.int32)
        for i, (candidates, role) in enumerate(izip(candidate_batch, roles)):
            for j, candidate in enumerate(candidates):
                T[i, j, 1:len(candidate)+1] = candidate
                # Insert <go> at the beginning at each turn because for decoding we want to
                # start from <go> to generate, except for padded turns
                if max_num_tokens == 0 or T[i][j][1] != int_markers.PAD:
                    T[i][j][0] = int_markers.GO_S if role == 'seller' else int_markers.GO_B
        return T

    def _create_candidate_batches(self):
        if self.dialogues[0].candidates is None:
            return None
        for dialogue in self.dialogues:
            assert len(dialogue.candidates) == self.num_turns
        candidate_batches = ([self._normalize_candidates(
            [dialogue.candidates[j] for dialogue in self.dialogues], [dialogue.role for dialogue in self.dialogues])
            for j in xrange(self.num_turns)])
        return candidate_batches

    def _create_turn_batches(self):
        turn_batches = []
        for i in xrange(Dialogue.num_stages):
            try:
                turn_batches.append([pad_list_to_array(
                    [d.turns[i][j] for d in self.dialogues], int_markers.PAD, np.int32)
                    for j in xrange(self.num_turns)])
            except IndexError:
                print 'num_turns:', self.num_turns
                for dialogue in self.dialogues:
                    print len(dialogue.turns[0]), len(dialogue.roles)
                import sys; sys.exit()
        return turn_batches

    def _create_price_batches(self):
        price_batches = [pad_list_to_array(
            [dialogue.price_turns[j] for dialogue in self.dialogues], int_markers.PAD, np.float32)
            for j in xrange(self.num_turns)]
        return price_batches

    def _create_context_batch(self):
        category_batch = np.array([d.category for d in self.dialogues], dtype=np.int32)
        # TODO: hacky
        pad = Dialogue.mappings['kb_vocab'].to_ind(markers.PAD)
        title_batch = pad_list_to_array([d.title for d in self.dialogues], pad, np.int32)
        description_batch = pad_list_to_array([d.description for d in self.dialogues], pad, np.int32)
        return {
                'category': category_batch,
                'title': title_batch,
                'description': description_batch,
                }

    def _get_agent_batch(self, i):
        return [dialogue.agents[i] for dialogue in self.dialogues]

    def _get_kb_batch(self):
        return [dialogue.kb for dialogue in self.dialogues]

    def pick_helpers(self, token_candidates, candidates):
        helpers = []
        for b, cands in enumerate(token_candidates):
            best_score = 0
            helper_id = -1
            for i, cand in enumerate(cands):
                try:
                    cand = cand['response']
                # Actual number of candidates can be smaller than num_candidates,
                # in which case an empty dict is returned instead
                except KeyError:
                    continue
                if i == 0:
                    target = cand
                    #print 'TARGET:', target
                else:
                    score = compute_bleu(cand, target)
                    #print 'CANDIDATE:', score, cand
                    if score > best_score:
                        best_score = score
                        helper_id = i
            if helper_id == -1:
                helper_id = 0
            #print 'HELPER:', helper_id, cands[helper_id]['response']
            helpers.append(candidates[b][helper_id])
        return np.array(helpers)

    def _mask_slots(self, targets):
        '''
        targets: target sequence (interger)
        return: mask (set to 1) words between <slot> and </slot> (including </slot> but not <slot>)
        '''
        mask = np.zeros_like(targets)
        for i, target in enumerate(targets):
            delimiters = np.where((target == int_markers.START_SLOT) | (target == int_markers.END_SLOT))[0]
            assert len(delimiters) % 2 == 0
            for j in xrange(0, delimiters.shape[0], 2):
                start, end = delimiters[j], delimiters[j+1]
                # Include </slot> but not <slot>
                mask[i][start+1:end+1] = 1
        targets[mask == 0] = int_markers.PAD
        return targets

    def _remove_last(self, array, value):
        nrows, ncols = array.shape
        for i in xrange(nrows):
            for j in xrange(ncols-1, -1, -1):
                if array[i][j] == value:
                    array[i][j] = int_markers.PAD
                    break
        return array

    def _create_one_batch(self, encode_turn, decode_turn, target_turn, price_encode_turn, price_decode_turn, encode_tokens, decode_tokens, token_candidates, candidates, agents, kbs, context_batch):
        if encode_turn is not None:
            # Remove <go> at the beginning of utterance
            encoder_inputs = encode_turn[:, 1:]
        else:
            raise ValueError
        # Remove pad (<go>) at the beginning of utterance
        encoder_price_inputs = price_encode_turn[:, 1:]

        # Decoder inputs: start from <go> to generate, i.e. <go> <token>
        assert decode_turn.shape == target_turn.shape
        decoder_inputs = np.copy(decode_turn)
        # NOTE: For each row, replace the last <eos> to <pad> to be consistent:
        # 1) The longest sequence does not have </s> as input.
        # 2) At inference time, decoder stops at </s>.
        decoder_inputs = self._remove_last(decoder_inputs, int_markers.EOS)[:, :-1]
        decoder_price_inputs = price_decode_turn[:, :-1]

        price_targets = price_decode_turn[:, 1:]
        decoder_targets = target_turn[:, 1:]

        # Mask for slots
        if self.slot_filling:
            decoder_targets = self._mask_slots(decoder_targets)

        #helpers = self.pick_helpers(token_candidates, candidates)  # (batch_size, helper_len)
        #context_batch['helper'] = helpers

        # TODO: group these
        batch = {
                 'encoder_inputs': encoder_inputs,
                 'decoder_inputs': decoder_inputs,
                 'targets': decoder_targets,
                 'decoder_price_inputs': decoder_price_inputs,
                 'encoder_price_inputs': encoder_price_inputs,
                 'price_targets': price_targets,
                 'encoder_tokens': encode_tokens,
                 'decoder_tokens': decode_tokens,
                 'token_candidates': token_candidates,
                 'candidates': candidates,
                 'agents': agents,
                 'kbs': kbs,
                 'context': context_batch,
                }
        return batch

    def _get_token_candidates(self, i):
        if self.dialogues[0].token_candidates is None:
            return None
        return [dialogue.token_candidates[i] if i < len(dialogue.token_candidates) else ''
                for dialogue in self.dialogues]

    def _get_token_turns(self, i):
        stage = 0
        if not hasattr(self.dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[i] if i < len(dialogue.token_turns) else ''
                for dialogue in self.dialogues]

    def create_batch(self):
        self._normalize_dialogue()
        turn_batches = self._create_turn_batches()  # (batch_size, num_turns)
        price_batches = self._create_price_batches()  # (batch_size, num_turns, price_feat_size)
        candidate_batches = self._create_candidate_batches()  # (batch_size, num_candidate, seq_len)

        enc, dec, tgt = Dialogue.ENC, Dialogue.DEC, Dialogue.TARGET

        # Add agents and kbs
        agents = self._get_agent_batch(1)  # Decoding agent
        kbs = self._get_kb_batch()

        context_batch = self._create_context_batch()

        # NOTE: when creating dialogue turns (see add_utterance), we have set the first utterance to be from the encoding agent
        encode_turn_ids = range(0, self.num_turns-1, 2)
        batch_seq = [self._create_one_batch(turn_batches[enc][i], turn_batches[dec][i+1], turn_batches[tgt][i+1], price_batches[i], price_batches[i+1], self._get_token_turns(i), self._get_token_turns(i+1), self._get_token_candidates(i+1), candidate_batches[i+1] if candidate_batches else None, agents, kbs, context_batch) for i in encode_turn_ids]

        # bath_seq: A sequence of batches should be processed in turn as the state of each batch is
        # passed on to the next batch
        batch = {
                 'batch_seq': batch_seq,
                }

        return batch

class LMDialogueBatch(DialogueBatch):
    def _create_one_batch_lm(self, tokens):
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        batch = {
                'inputs': inputs,
                'targets': targets,
                }
        return batch

    def _create_one_batch(self, encode_turn, decode_turn, target_turn, price_encode_turn, price_decode_turn, encode_tokens, decode_tokens, token_candidates, candidates, agents, kbs, context_batch):
        encoder_inputs = encode_turn

        # Remove pad (<go>) at the beginning of utterance
        #encoder_price_inputs = price_encode_turn[:, 1:]

        # Decoder inputs: start from <go> to generate, i.e. <go> <token>
        assert decode_turn.shape == target_turn.shape
        decoder_inputs = np.copy(decode_turn)

        # Include pad (<go>) at the beginning of utterance
        #decoder_price_inputs = price_decode_turn[:, :-1]
        #price_targets = price_decode_turn[:, 1:]

        decoder_targets = target_turn[:, 1:]

        #helpers = self.pick_helpers(token_candidates, candidates)  # (batch_size, helper_len)
        #context_batch['helper'] = helpers

        # TODO: group these
        batch = {
                 'encoder_inputs': encoder_inputs,
                 'decoder_inputs': decoder_inputs,
                 'targets': decoder_targets,
                 'encoder_tokens': encode_tokens,
                 'decoder_tokens': decode_tokens,
                 'token_candidates': token_candidates,
                 'candidates': candidates,
                 'agents': agents,
                 'kbs': kbs,
                 'context': context_batch,
                }
        return batch

    def create_batch(self, bptt_steps):
        lm_batch = self._create_lm_batch(bptt_steps)
        batch = super(LMDialogueBatch, self).create_batch()
        context = batch['batch_seq'][0]['context']
        batch['eval_batch_seq'] = batch['batch_seq']
        for b in lm_batch:
            b['context'] = context
        batch['batch_seq'] = lm_batch
        return batch

    def _create_lm_batch(self, bptt_steps=35):
        data = [d.join_turns() for d in self.dialogues]
        dialogue_tokens = [d[0] for d in data]
        # TODO: don't need dialogue_turns
        dialogue_turns = [d[1] for d in data]
        max_len = max([len(tokens) for tokens in dialogue_tokens])
        batch_size = len(self.dialogues)
        T = np.full([batch_size, max_len], int_markers.PAD, dtype=np.int32)
        for i, tokens in enumerate(dialogue_tokens):
            T[i, :len(tokens)] = tokens

        batch_seq = []
        for i in range(0, max_len, bptt_steps):
            if i + bptt_steps > max_len - 5:
                batch_seq.append(self._create_one_batch_lm(T[:, i:]))
                break
            else:
                batch_seq.append(self._create_one_batch_lm(T[:, i:i+bptt_steps]))

        return batch_seq


class Preprocessor(object):
    '''
    Preprocess raw utterances: tokenize, entity linking.
    Convert an Example into a Dialogue data structure used by DataGenerator.
    '''
    def __init__(self, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form, slot_filling=False, slot_detector=None):
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
        if stage is None:
            return [self.get_entity_form(x, 'canonical') if is_entity(x) else x for x in utterance]
        else:
            return [self.get_entity_form(x, self.entity_forms[stage]) if is_entity(x) else x for x in utterance]

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue from each agent's perspective
        Create two Dialogue objects for each example
        '''
        kbs = ex.scenario.kbs
        for agent in (0, 1):
            dialogue = Dialogue(agent, kbs[agent], ex.ex_id)
            for e in ex.events:
                utterance = self.process_event(e, dialogue.agent, dialogue.kb)
                if utterance:
                    dialogue.add_utterance(e.agent, utterance)
            yield dialogue

    @classmethod
    def price_to_entity(cls, price):
        #return (price, (price, 'price'))
        return Entity(price, CanonicalEntity(price, 'price'))

    def _mark_slots(self, utterance):
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

    def process_event(self, e, agent, kb):
        '''
        Tokenize, link entities
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.link_entity(tokenize(e.data), kb=kb)
            if self.slot_filling:
                entity_tokens = self.slot_detector.detect_slots(entity_tokens, kb=kb, join=False)
                entity_tokens = self._mark_slots(entity_tokens)
            if entity_tokens:
                return entity_tokens
            else:
                return None
        elif e.action == 'offer':
            data = e.data['price']
            entity_tokens = [markers.OFFER, self.price_to_entity(data)]
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
    def __init__(self, train_examples, dev_examples, test_examples, preprocessor, schema, mappings=None, retriever=None, cache='.cache', ignore_cache=False, candidates_path=[]):
        examples = {'train': train_examples or [], 'dev': dev_examples or [], 'test': test_examples or []}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.iteritems()}

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
            self.dialogues = {k: preprocessor.preprocess(v)  for k, v in examples.iteritems()}

            for fold, dialogues in self.dialogues.iteritems():
                print '%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold])
        else:
            print 'Using cached data from', cache

        if not mappings:
            mappings = create_mappings(self.dialogues['train'], schema, preprocessor.entity_forms.values())
        self.mappings = mappings

        self.textint_map = TextIntMap(mappings['vocab'], preprocessor)
        Dialogue.mappings = mappings
        Dialogue.textint_map = self.textint_map
        Dialogue.preprocessor = preprocessor

        global int_markers
        int_markers = SpecialSymbols(*[mappings['vocab'].to_ind(m) for m in markers])

    def convert_to_int(self):
        '''
        Convert tokens to integers.
        '''
        for fold, dialogues in self.dialogues.iteritems():
            for dialogue in dialogues:
                dialogue.convert_to_int()

    def create_dialogue_batches(self, dialogues, batch_size):
        dialogue_batches = []
        dialogues.sort(key=lambda d: len(d.turns[0]))
        N = len(dialogues)
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.append(DialogueBatch(dialogue_batch, self.slot_filling).create_batch())
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
        tokens = candidate.get('response', [])
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
                    new_cand = dict(candidate)
                    new_cand['response'] = new_tokens
                    rewritten.append(new_cand)
        return rewritten

    def generator(self, name, batch_size, shuffle=True):
        if not os.path.isdir(self.cache):
            os.makedirs(self.cache)
        cache_file = os.path.join(self.cache, '%s_batches.pkl' % name)
        if (not os.path.exists(cache_file)) or self.ignore_cache:
            dialogues = self.dialogues[name]

            if self.retriever is not None:
                Dialogue.num_candidates = 50
                for dialogue in dialogues:
                    k = (dialogue.uuid, dialogue.role)
                    # candidates: list of list containing num_candidates responses for each turn of the dialogue
                    # None candidates means that it is not the agent's speaking turn
                    if k in self.cached_candidates:
                        candidates = self.cached_candidates[k]
                    else:
                        candidates = self.retriever.retrieve_candidates(dialogue, json_dict=False)

                    candidates = [c if c else self.retriever.empty_candidates for c in candidates]

                    kb_context = self.slot_detector.context_to_fillers(dialogue.kb)
                    new_candidates = []
                    for i, turn_candidates in enumerate(candidates):
                        rewritten_cands = []
                        for candidate in turn_candidates:
                            rewritten_cands.append(candidate)
                            rewritten = self.rewrite_candidate(kb_context, candidate)
                            rewritten_cands.extend(rewritten)
                        new_candidates.append(rewritten_cands)
                    for i, cands in enumerate(new_candidates):
                        if len(cands) < Dialogue.num_candidates:
                            cands.extend([{} for _ in xrange(Dialogue.num_candidates - len(cands))])
                        else:
                            new_candidates[i] = cands[:Dialogue.num_candidates]

                    dialogue.add_candidates(new_candidates)
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
            print 'Read %d batches from cache %s' % (len(dialogue_batches), cache_file)
            print '[%d s]' % (time.time() - start_time)

        yield len(dialogue_batches)
        inds = range(len(dialogue_batches))
        while True:
            if shuffle:
                random.shuffle(inds)
            for ind in inds:
                yield dialogue_batches[ind]

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
            dialogue_batches.append(LMDialogueBatch(dialogue_batch).create_batch(bptt_steps))
            start = end
        return dialogue_batches

