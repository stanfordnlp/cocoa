'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from src.model.vocab import Vocabulary
from src.basic.entity import Entity, CanonicalEntity, is_entity
from itertools import chain, izip
from collections import namedtuple, defaultdict
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
        #offset = self.vocab.size
        #return [(get_price(token) + offset) if get_price(token) is not None else self.vocab.to_ind(token) for token in tokens]
        return [self.vocab.to_ind(token) for token in tokens]

    def process_entity(self, token_array, stage):
        '''
        token_array: 2D int array of tokens, assuming entities are mapped by entity_map offset by vocab_size.
        Transform entities to specified form, e.g. type, canonical form.
        Return the new token_array and an entity_array which has the same shape as token_array and use -1
        for non-entity words and map entities by entity_map.
        '''
        offset = self.vocab.size
        entity_array = np.full(token_array.shape, -1, dtype=np.float32)
        entity_inds = token_array >= offset
        entity_array[entity_inds] = token_array[entity_inds] - offset

        # Entities needs to be transformed and mapped by vocab
        for tok_int, is_ent in izip(np.nditer(token_array, op_flags=['readwrite']), np.nditer(entity_inds)):
            if is_ent:
                price = tok_int[0]  # This is actually float for prices
                # NOTE: at this point we have lost the surface form of the entity: using an empty string
                processed_entity = self.preprocessor.get_entity_form(('', (price, 'price')), self.entity_forms[stage])
                tok_int[...] = self.vocab.to_ind(processed_entity)
        return token_array, entity_array

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

    def __init__(self, agent, kb, uuid):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        # token_turns: tokens and entitys (output of entitylink)
        self.token_turns = []
        # entities: -1 for non-entity words, entities are mapped based on entity_map
        self.entities = ([], [])
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = ([], [], [])
        #self.prices = []
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
        #prices = [x if (is_entity(x) and x[1][1] == 'price') else None for x in utterance]
        utterance = self.normalize_price(self.kb, utterance)
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            self.token_turns[-1].append(utterance)
        else:
            self.agents.append(agent)
            self.roles.append(self.kb.facts['personal']['Role'])
            self.token_turns.append([utterance])
            #self.prices.append([prices])

    def convert_to_int(self):
        if self.is_int:
            return
        for turn in self.token_turns:
            for i in xrange(len(self.turns)):
                self.turns[i].append([self.textint_map.text_to_int(utterance)
                    for utterance in turn])
        self.is_int = True

    def flatten_turns(self):
        '''
        Flatten turns to a list of tokens with </s> between utterances and </t> at the end.
        '''
        if self.flattened:
            return

        for i, turns in enumerate(self.turns):
            for j, turn in enumerate(turns):
                for utterance in turn:
                    utterance.append(int_markers.EOS)
                self.turns[i][j] = [x for x in chain.from_iterable(turn)]

        if hasattr(self, 'token_turns'):
            for j, turn in enumerate(self.token_turns):
                for utterance in turn:
                    utterance.append(markers.EOS)
                self.token_turns[j] = [x for x in chain.from_iterable(turn)]

        self.flattened = True

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        turns, agents, roles = self.turns, self.agents, self.roles
        assert len(turns[0]) == len(turns[1]) and len(turns[0]) == len(turns[2])
        for i in xrange(len(turns[0]), num_turns):
            #agents.append(1 - agents[-1])
            agents.append(None)
            roles.append(None)
            for j in xrange(len(turns)):
                turns[j].append([])

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

    def _get_agent_batch(self, i):
        return [dialogue.agents[i] for dialogue in self.dialogues]

    def _get_kb_batch(self):
        return [dialogue.kb for dialogue in self.dialogues]

    def _remove_last(self, array, value):
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

    def _create_one_batch(self, encode_turn, decode_turn, target_turn, encode_tokens, decode_tokens, agents, kbs):
        if encode_turn is not None:
            # Remove <go> at the beginning of utterance
            encoder_inputs = encode_turn[:, 1:]
        else:
            batch_size = decode_turn.shape[0]
            # If there's no input to encode, use </s> as the encoder input.
            encoder_inputs = np.full((batch_size, 1), int_markers.EOS, dtype=np.int32)
            # encode_tokens are empty lists
            encode_tokens = [[''] for _ in xrange(batch_size)]

        # Decoder inputs: start from <go> to generate, i.e. <go> <token>
        assert decode_turn.shape == target_turn.shape
        decoder_inputs = np.copy(decode_turn)
        decoder_inputs = self._remove_last(decoder_inputs, int_markers.EOS)[:, :-1]
        decoder_targets = target_turn[:, 1:]

        # Process entities. NOTE: change turns in place!
        # prices to '<price>'
        #encoder_inputs, encoder_prices  = Dialogue.textint_map.process_entity(encoder_inputs, 'encoding')
        #decoder_inputs, decoder_prices = Dialogue.textint_map.process_entity(decoder_inputs, 'decoding')
        #decoder_targets, price_targets = Dialogue.textint_map.process_entity(decoder_targets, 'target')

        # TODO: use textint_map to process encoder/decoder_inputs here
        batch = {
                 'encoder_inputs': encoder_inputs,
                 'decoder_inputs': decoder_inputs,
                 'targets': decoder_targets,
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
        # TODO: create_price_batches
        # A sequence of batches should be processed in turn as the state of each batch is
        # passed on to the next batch
        batches = []
        enc, dec, tgt = Dialogue.ENC, Dialogue.DEC, Dialogue.TARGET

        # Add agents and kbs
        agents = self._get_agent_batch(1)  # Decoding agent
        kbs = self._get_kb_batch()

        # NOTE: when creating dialogue turns (see add_utterance), we have set the first utterance to be from the encoding agent
        encode_turn_ids = range(0, self.num_turns-1, 2)
        batch_seq = [self._create_one_batch(turn_batches[enc][i], turn_batches[dec][i+1], turn_batches[tgt][i+1], self._get_token_turns(i), self._get_token_turns(i+1), agents, kbs) for i in encode_turn_ids]

        batch = {
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
    def __init__(self, train_examples, dev_examples, test_examples, preprocessor, schema, mappings=None, use_kb=False, copy=False):
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
