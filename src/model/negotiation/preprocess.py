'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from src.model.vocab import Vocabulary
from src.basic.entity import Entity, CanonicalEntity, is_entity
from itertools import chain, izip
from collections import namedtuple, defaultdict, deque
import copy
from nltk.tokenize import word_tokenize
from src.basic.negotiation.price_tracker import PriceTracker
get_price = PriceTracker.get_price

def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['type', 'canonical'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'type'], default='canonical', help='Output entity form to the decoder')

SpecialSymbols = namedtuple('SpecialSymbols', ['EOS', 'GO_S', 'GO_B', 'OFFER', 'QUIT', 'PAD'])
markers = SpecialSymbols(EOS='</s>', GO_S='<go-s>', GO_B='<go-b>', OFFER='<offer>', QUIT='<quit>', PAD='<pad>')
START_PRICE = -1

def tokenize(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    tokens = word_tokenize(utterance)
    # Don't split on markers <>
    new_tokens = []
    in_brackets = False
    for tok in tokens:
        if in_brackets:
            new_tokens[-1] = new_tokens[-1] + tok
        else:
            new_tokens.append(tok)
        if tok == '<':
            in_brackets = True
        if tok == '>':
            in_brackets = False
    return new_tokens

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
            for token in chain.from_iterable(turn):
                if is_entity(token):
                    _add_entity(token)
                else:
                    vocab.add_word(token)

    # Add special symbols
    vocab.add_words(special_symbols)
    print 'Vocabulary size:', vocab.size
    return vocab

def create_mappings(dialogues, schema, entity_forms):
    vocab = build_vocab(dialogues, markers, entity_forms)
    return {'vocab': vocab,
            }

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
        return [self.vocab.to_word(ind) for ind in inds]

class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target
    price_hist_len = 3

    def __init__(self, agent, kb, uuid):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        # token_turns: tokens and entitys (output of entitylink)
        self.token_turns = []
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = [[], [], []]
        # entities: has the same structure as turns, non-entity tokens are None
        self.entities = []
        self.agents = []
        self.roles = []
        self.is_int = False  # Whether we've converted it to integers
        self.flattened = False

    def add_utterance(self, agent, utterances):
        # Always start from the partner agent
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [])
        self._add_utterance(agent, utterances)

    @classmethod
    # TODO: this is operated on canonical entities, need to be consistent!
    def _original_price(cls, kb, price):
        p = get_price(price)
        b = kb.facts['personal']['Bottomline']  # 0
        t = kb.facts['personal']['Target']  # 1
        w = 1. / (t - b)
        c = -1. * b / (t - b)
        assert w != 0
        p = (p - c) / w
        p = int(p)
        return price._replace(canonical=price.canonical._replace(value=p))

    @classmethod
    def _normalize_price(cls, kb, price):
        '''
        Normalize the price such that bottomline=0 and target=1.
        '''
        p = get_price(price)
        b = kb.facts['personal']['Bottomline']  # 0
        t = kb.facts['personal']['Target']  # 1
        assert (t - b) != 0
        w = 1. / (t - b)
        c = -1. * b / (t - b)
        p = w * p + c
        p = float('{:.2f}'.format(p))
        return price._replace(canonical=price.canonical._replace(value=p))

    @classmethod
    def normalize_price(cls, kb, utterance):
        return [cls._normalize_price(kb, x) if is_entity(x) else x for x in utterance]

    @classmethod
    def original_price(cls, kb, utterance):
        s = [cls._original_price(kb, x) if is_entity(x) else x for x in utterance]
        return s

    def _add_utterance(self, agent, utterance):
        utterance = self.normalize_price(self.kb, utterance)
        entities = [x if is_entity(x) else None for x in utterance]
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            self.token_turns[-1].append(utterance)
            self.entities[-1].append(entities)
        else:
            self.agents.append(agent)
            self.roles.append(self.kb.facts['personal']['Role'])
            self.token_turns.append([utterance])
            self.entities.append([entities])

    def convert_to_int(self):
        if self.is_int:
            return
        for turn in self.token_turns:
            for turns, stage in izip(self.turns, ('encoding', 'decoding', 'target')):
                turns.append([self.textint_map.text_to_int(utterance, stage)
                    for utterance in turn])
        self.is_int = True

    def _flatten_turns(self, turns, EOS):
        flat_turns = []
        for turn in turns:
            for utterance in turn:
                utterance.append(EOS)
            flat_turns.append([x for x in chain.from_iterable(turn)])
        return flat_turns

    def flatten_turns(self):
        '''
        Flatten turns to a list of tokens with </s> between utterances and </t> at the end.
        '''
        if self.flattened:
            return

        for i, turns in enumerate(self.turns):
            self.turns[i] = self._flatten_turns(turns, int_markers.EOS)
        self.token_turns = self._flatten_turns(self.token_turns, markers.EOS)
        self.entities = self._flatten_turns(self.entities, None)
        self.price_hists = self.get_price_hist(self.price_hist_len, int_markers.PAD)

        self.flattened = True

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
        # TODO: we need a price pad vector, currently this is hard-coded, move to a class
        self.price_hists = self._pad_list(self.price_hists, num_turns, [[int_markers.PAD]*8])
        assert len(self.price_hists) == len(self.turns[0])

    def get_price_hist(self, hist_len, pad):
        '''
        Given flattened entity turns, return the entity history for each token.
        pad: used to fill in non-price targets.
        '''
        my_prices = deque([START_PRICE] * hist_len, maxlen=hist_len)
        partner_prices = deque([START_PRICE] * hist_len, maxlen=hist_len)
        price_hists = []
        assert len(self.agents) == len(self.entities)
        for agent, entities in izip(self.agents, self.entities):
            h = []
            if agent == self.agent:
                price_queue = my_prices
            else:
                price_queue = partner_prices
            # This is a padded turn at the beginning
            if len(entities) == 0:
                price_vec = [pad, agent] + list(my_prices) + list(partner_prices)
                h.append(price_vec)
            else:
                for entity in entities:
                    if entity is not None:
                        p = float('{:.2f}'.format(get_price(entity)))
                        price_queue.append(p)
                    else:
                        p = pad
                    # NOTE: we put the current price p at the begining, will be sliced to be the price target later
                    price_vec = [p, agent] + list(my_prices) + list(partner_prices)
                    h.append(price_vec)
            assert len(h) > 0
            price_hists.append(h)
        return price_hists


class DialogueBatch(object):
    use_kb = False
    copy = False

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def _normalize_dialogue(self):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([len(d.turns[0]) for d in self.dialogues])
        for dialogue in self.dialogues:
            dialogue.flatten_turns()
            dialogue.pad_turns(max_num_turns)
        self.num_turns = len(self.dialogues[0].turns[0])

    @classmethod
    def _normalize_turn(cls, turn_batch, roles):
        '''
        All turns at the same time step should have the same number of tokens.
        '''
        max_num_tokens = max([len(t) for t in turn_batch])
        batch_size = len(turn_batch)
        T = np.full([batch_size, max_num_tokens+1], int_markers.PAD, dtype=np.int32)
        for i, (turn, role) in enumerate(izip(turn_batch, roles)):
            T[i, 1:len(turn)+1] = turn
            # Insert <go> at the beginning at each turn because for decoding we want to
            # start from <go> to generate, except for padded turns
            if T[i][1] != int_markers.PAD:
                T[i][0] = int_markers.GO_S if role == 'seller' else int_markers.GO_B
        return T

    @classmethod
    def _normalize_price_hist(cls, price_batch, roles):
        '''
        Perform the same ops as _normalize turn: pad price hist for padded tokens.
        '''
        max_num_tokens = max([len(t) for t in price_batch])
        batch_size = len(price_batch)
        feat_size = len(price_batch[0][0])
        T = np.full([batch_size, max_num_tokens+1, feat_size], START_PRICE, dtype=np.float32)
        for i, (turn, role) in enumerate(izip(price_batch, roles)):
            T[i, 1:len(turn)+1, :] = turn
        return T

    def _create_turn_batches(self):
        turn_batches = []
        for i in xrange(Dialogue.num_stages):
            try:
                turn_batches.append([self._normalize_turn(
                    [dialogue.turns[i][j] for dialogue in self.dialogues], [dialogue.roles[j] for dialogue in self.dialogues])
                    for j in xrange(self.num_turns)])
            except IndexError:
                print 'num_turns:', self.num_turns
                for dialogue in self.dialogues:
                    print len(dialogue.turns[0]), len(dialogue.roles)
                import sys; sys.exit()
        return turn_batches

    def _create_price_batches(self):
        price_batches = [self._normalize_price_hist(
            [dialogue.price_hists[j] for dialogue in self.dialogues], [dialogue.roles[j] for dialogue in self.dialogues])
            for j in xrange(self.num_turns)]
        return price_batches


    def _get_agent_batch(self, i):
        return [dialogue.agents[i] for dialogue in self.dialogues]

    def _get_kb_batch(self):
        return [dialogue.kb for dialogue in self.dialogues]

    def _replace_eos(self, array, value):
        '''
        For each row, replace (in place) the last occurence of value to <pad>.
        The last token input to decoder should not be </s> otherwise the model will learn
        </s> <pad> (deterministically).
        '''
        nrows, ncols = array.shape
        for i in xrange(nrows):
            for j in xrange(ncols-1, -1, -1):
                if array[i][j] == value:
                    array[i][j] = int_markers.PAD
                    break
        return array

    def _create_one_batch(self, encode_turn, decode_turn, target_turn, price_turn, encode_tokens, decode_tokens, agents, kbs):
        if encode_turn is not None:
            # Remove <go> at the beginning of utterance
            encoder_inputs = encode_turn[:, 1:]
        else:
            raise ValueError
            batch_size = decode_turn.shape[0]
            # If there's no input to encode, use </s> as the encoder input.
            encoder_inputs = np.full((batch_size, 1), int_markers.EOS, dtype=np.int32)
            # encode_tokens are empty lists
            encode_tokens = [[''] for _ in xrange(batch_size)]

        # Decoder inputs: start from <go> to generate, i.e. <go> <token>
        assert decode_turn.shape == target_turn.shape
        decoder_inputs = np.copy(decode_turn)
        decoder_inputs = self._replace_eos(decoder_inputs, int_markers.EOS)[:, :-1]
        decoder_price_inputs = price_turn[:, :-1, 1:]

        decoder_targets = target_turn[:, 1:]
        price_targets = price_turn[:, 1:, 0]

        # TODO: use textint_map to process encoder/decoder_inputs here
        batch = {
                 'encoder_inputs': encoder_inputs,
                 'decoder_inputs': decoder_inputs,
                 'targets': decoder_targets,
                 'decoder_price_inputs': decoder_price_inputs,
                 'price_targets': price_targets,
                 'encoder_tokens': encode_tokens,
                 'decoder_tokens': decode_tokens,
                 'agents': agents,
                 'kbs': kbs,
                }
        return batch

    def _get_token_turns(self, i):
        stage = 0
        if not hasattr(self.dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[i] if i < len(dialogue.token_turns) else ''
                for dialogue in self.dialogues]

    def create_batches(self):
        self._normalize_dialogue()
        turn_batches = self._create_turn_batches()  # (batch_size, num_turns)
        price_batches = self._create_price_batches()  # (batch_size, num_turns, price_feat_size)
        # A sequence of batches should be processed in turn as the state of each batch is
        # passed on to the next batch
        batches = []
        enc, dec, tgt = Dialogue.ENC, Dialogue.DEC, Dialogue.TARGET

        # Add agents and kbs
        agents = self._get_agent_batch(1)  # Decoding agent
        kbs = self._get_kb_batch()

        # NOTE: when creating dialogue turns (see add_utterance), we have set the first utterance to be from the encoding agent
        encode_turn_ids = range(0, self.num_turns-1, 2)
        batch_seq = [self._create_one_batch(turn_batches[enc][i], turn_batches[dec][i+1], turn_batches[tgt][i+1], price_batches[i+1], self._get_token_turns(i), self._get_token_turns(i+1), agents, kbs) for i in encode_turn_ids]

        batch = {
                 'agent': agents,
                 'kb': kbs,
                 'batch_seq': batch_seq,
                }
        batches.append(batch)

        return batches

class Preprocessor(object):
    '''
    Preprocess raw utterances: tokenize, entity linking.
    Convert an Example into a Dialogue data structure used by DataGenerator.
    '''
    def __init__(self, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form):
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon
        self.entity_forms = {'encoding': entity_encoding_form,
                'decoding': entity_decoding_form,
                'target': entity_target_form}

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
            return entity.canonical
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
            dialogue = Dialogue(agent, kbs[agent], ex.uuid)
            mentioned_entities = set()
            for e in ex.events:
                utterance = self.process_event(e, dialogue.agent, dialogue.kb, mentioned_entities)
                if utterance:
                    dialogue.add_utterance(e.agent, utterance)
                    for token in utterance:
                        if is_entity(token):
                            mentioned_entities.add(token.canonical)
            yield dialogue

    @classmethod
    def price_to_entity(cls, price):
        #return (price, (price, 'price'))
        return Entity(price, CanonicalEntity(price, 'price'))

    def process_event(self, e, agent, kb, mentioned_entities=None):
        '''
        Convert event to two lists of tokens and entities for encoding and decoding.
        agent: from this agent's perspective
        kb: the agent's (known) kb
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            if agent == e.agent:
                entity_tokens = self.lexicon.link_entity(tokenize(e.data), kb=kb.facts, mentioned_entities=mentioned_entities)
            else:
                entity_tokens = self.lexicon.link_entity(tokenize(e.data), partner_kb=kb.facts, mentioned_entities=mentioned_entities)
            #entity_tokens = [x if not is_entity(x) else x for x in entity_tokens]
            if entity_tokens:
                return entity_tokens
            else:
                return None
        elif e.action == 'offer':
            entity_tokens = [markers.OFFER, self.price_to_entity(e.data)]
            return entity_tokens
        elif e.action == 'quit':
            entity_tokens = [markers.QUIT]
            return entity_tokens
        else:
            raise ValueError('Unknown event action.')

    def preprocess(self, examples):
        dialogues = []
        for ex in examples:
            for d in self._process_example(ex):
                # Skip incomplete chats
                if len(d.agents) < 2 or ex.outcome['reward'] == 0:
                    continue
                    print 'Removing dialogue %s' % d.uuid
                    for event in ex.events:
                        print event.to_dict()
                else:
                    dialogues.append(d)
        return dialogues

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, preprocessor, schema, mappings=None, use_kb=False, copy=False, price_hist_len=3):
        examples = {'train': train_examples or [], 'dev': dev_examples or [], 'test': test_examples or []}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.iteritems()}
        self.use_kb = use_kb  # Whether to generate graph
        self.copy = copy

        DialogueBatch.use_kb = use_kb
        DialogueBatch.copy = copy

        # NOTE: each dialogue is made into two examples from each agent's perspective
        self.dialogues = {k: preprocessor.preprocess(v)  for k, v in examples.iteritems()}

        for fold, dialogues in self.dialogues.iteritems():
            print '%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold])

        if not mappings:
            mappings = create_mappings(self.dialogues['train'], schema, preprocessor.entity_forms.values())
        self.mappings = mappings

        self.textint_map = TextIntMap(mappings['vocab'], preprocessor)
        Dialogue.textint_map = self.textint_map
        Dialogue.preprocessor = preprocessor
        Dialogue.price_hist_len = 3

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
            dialogue_batches.extend(DialogueBatch(dialogue_batch).create_batches())
            start = end
        return dialogue_batches

    def generator(self, name, batch_size, shuffle=True):
        dialogues = self.dialogues[name]
        for dialogue in dialogues:
            dialogue.convert_to_int()
        dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)
        yield len(dialogue_batches)
        inds = range(len(dialogue_batches))
        while True:
            if shuffle:
                random.shuffle(inds)
            for ind in inds:
                yield dialogue_batches[ind]
