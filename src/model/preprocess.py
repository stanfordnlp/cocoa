'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from src.model.vocab import Vocabulary, is_entity
from src.model.graph import Graph, GraphBatch, inv_rel
from itertools import chain, izip
from collections import namedtuple, defaultdict
import copy

def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['surface', 'type', 'canonical'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'graph'], default='canonical', help='Output entity form to the decoder')

SpecialSymbols = namedtuple('SpecialSymbols', ['EOS', 'GO', 'SELECT', 'PAD'])
markers = SpecialSymbols(EOS='</s>', GO='<go>', SELECT='<select>', PAD='<pad>')

def tokenize(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    # Split on punctuation
    tokens = re.findall(r"[\w']+|[.,!?;]", utterance)
    # Remove punctuation
    tokens = [x for x in tokens if x not in '.,!?;']
    return tokens

def build_schema_mappings(schema, num_items):
    entity_map = Vocabulary(unk=True)
    for type_, values in schema.values.iteritems():
        entity_map.add_words(((value.lower(), type_) for value in values))
    # Add item nodes
    for i in xrange(num_items):
        entity_map.add_word(('item-%d' % i, 'item'))
    # TODO: add attribute nodes

    relation_map = Vocabulary(unk=False)
    attribute_types =  schema.get_attributes()  # {attribute_name: value_type}
    relation_map.add_words((a.lower() for a in attribute_types.keys()))
    relation_map.add_word('has')
    # Inverse relation
    relation_map.add_words([inv_rel(r) for r in relation_map.word_to_ind])

    return entity_map, relation_map

def build_vocab(dialogues, special_symbols=[], entity_forms=[]):
    vocab = Vocabulary(offset=0, unk=True)

    def _add_entity(entity):
        for entity_form in entity_forms:
            # If copy entity embedding from the graph embedding, don't need entity in vocab
            if entity_form != 'graph':
                word = Preprocessor.get_entity_form(entity, entity_form)
                vocab.add_word(word)

    # Add words
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for turns in dialogue.token_turns:
            for turn in turns:
                for token in chain.from_iterable(turn):
                    if is_entity(token):
                        _add_entity(token)
                    else:
                        vocab.add_word(token)

    # Add special symbols
    vocab.add_words(special_symbols)
    return vocab

def create_mappings(dialogues, schema, num_items, entity_forms):
    vocab = build_vocab(dialogues, markers, entity_forms)
    entity_map, relation_map = build_schema_mappings(schema, num_items)
    return {'vocab': vocab,
            'entity': entity_map,
            'relation': relation_map,
            }

class TextIntMap(object):
    '''
    Map between text and int for visualizing results.
    '''
    def __init__(self, vocab, entity_map, preprocessor):
        self.vocab = vocab
        self.entity_map = entity_map
        self.entity_forms = preprocessor.entity_forms
        self.preprocessor = preprocessor
        self.setting = {k: self.use_entity_map(v) for k, v in self.entity_forms.iteritems()}

    def pred_to_input(self, preds):
        '''
        Convert decoder outputs to decoder inputs.
        '''
        if self.entity_forms['target'] == self.entity_forms['decoding']:
            return preds
        preds_utterances = [self.int_to_text(pred, 'target') for pred in preds]
        input_utterances = [self.preprocessor.process_utterance(utterance, 'decoding') for utterance in preds_utterances]
        inputs = np.array([self.text_to_int(utterance, 'decoding') for utterance in input_utterances])
        return inputs

    def use_entity_map(self, entity_form):
        if entity_form == 'graph':
            return True
        return False

    def text_to_int(self, utterance, stage):
        '''
        Process entities in the utterance based on whether it is used for encoding, decoding
        or ground truth.
        '''
        use_entity_map = self.setting[stage]
        tokens = self.preprocessor.process_utterance(utterance, stage)
        if not use_entity_map:
            return [self.vocab.to_ind(token) if not is_entity(token) else self.vocab.to_ind(token) for token in tokens]
        else:
            offset = self.vocab.size
            return [self.vocab.to_ind(token) if not is_entity(token) else self.entity_map.to_ind(token) + offset for token in tokens]

    def int_to_text(self, inds, stage):
        '''
        Inverse of text_to_int.
        '''
        use_entity_map = self.setting[stage]
        if not use_entity_map:
            return [self.vocab.to_word(ind) for ind in inds]
        else:
            return [self.vocab.to_word(ind) if ind < self.vocab.size else self.entity_map.to_word(ind - self.vocab.size) for ind in inds]

class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target

    def __init__(self, kbs, uuid):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.kbs = kbs
        # token_turns: tokens and entitys (output of entitylink)
        self.token_turns = ([], [])
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = ([], [], [])
        self.agents = []
        self.is_int = False  # Whether we've converted it to integers
        self.flattened = False

    def create_graph(self):
        assert not hasattr(self, 'graphs')
        self.graphs = [Graph(kb) for kb in self.kbs]

    def add_utterance(self, agent, utterances):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            for i in xrange(2):
                self.token_turns[i][-1].append(utterances[i])
        else:
            self.agents.append(agent)
            for i in xrange(2):
                self.token_turns[i].append([utterances[i]])

    def convert_to_int(self):
        if self.is_int:
            return
        for i, turns in enumerate(self.token_turns):
            if i == self.ENC:
                stage = 'encoding'
            elif i == self.DEC:
                stage = 'decoding'
            else:
                raise ValueError('Unknown stage %s' % stage)
            for turn in turns:
                self.turns[i].append([self.textint_map.text_to_int(utterance, stage)
                    for utterance in turn])
        # Target tokens
        stage = 'target'
        for turn in self.token_turns[self.DEC]:
            self.turns[self.TARGET].append([self.textint_map.text_to_int(utterance, stage)
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
            for i, turns in enumerate(self.token_turns):
                for j, turn in enumerate(turns):
                    self.token_turns[i][j] = [x for x in chain.from_iterable(turn)]

        self.flattened = True

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        turns, agents = self.turns, self.agents
        assert len(turns[0]) == len(turns[1]) and len(turns[0]) == len(turns[2])
        for i in xrange(len(turns[0]), num_turns):
            agents.append(1 - agents[-1])
            for j in xrange(len(turns)):
                turns[j].append([])

class DialogueBatch(object):
    def __init__(self, dialogues, use_kb=False):
        self.dialogues = dialogues
        self.use_kb = use_kb

    def _normalize_dialogue(self):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([len(d.turns[0]) for d in self.dialogues])
        for dialogue in self.dialogues:
            dialogue.flatten_turns()
            dialogue.pad_turns(max_num_turns)
        self.num_turns = len(self.dialogues[0].turns[0])

    def _normalize_turn(self, turn_batch):
        '''
        All turns at the same time step should have the same number of tokens.
        '''
        max_num_tokens = max([len(t) for t in turn_batch])
        batch_size = len(turn_batch)
        T = np.full([batch_size, max_num_tokens+1], int_markers.PAD, dtype=np.int32)
        for i, turn in enumerate(turn_batch):
            T[i, 1:len(turn)+1] = turn
            # Insert <go> at the beginning at each turn because for decoding we want to
            # start from <go> to generate, except for padded turns
            if T[i][1] != int_markers.PAD:
                T[i][0] = int_markers.GO
        return T

    def _create_turn_batches(self):
        turn_batches = []
        for i in xrange(Dialogue.num_stages):
            turn_batches.append([self._normalize_turn(
                [dialogue.turns[i][j] for dialogue in self.dialogues])
                for j in xrange(self.num_turns)])
        return turn_batches

    def _get_agent_batch(self, i):
        return [dialogue.agents[i] for dialogue in self.dialogues]

    def _get_kb_batch(self, agents):
        return [dialogue.kbs[agent] for dialogue, agent in izip(self.dialogues, agents)]

    def _get_graph_batch(self, agents):
        return GraphBatch([dialogue.graphs[agent] for dialogue, agent in izip(self.dialogues, agents)])

    def _remove_last(self, array, value):
        '''
        For each row, replace (in place) the last occurence of value to <pad>.
        The last token input to decoder should not be </s> otherwise the model will learn
        </s> <pad> (deterministically).
        '''
        col_inds = np.max(np.where(array == value), axis=1)
        row_inds = np.arange(array.shape[0])
        array[row_inds, col_inds] = int_markers.PAD
        return array

    def _create_one_batch(self, encode_turn, decode_turn, target_turn, encode_tokens, decode_tokens):
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
        decoder_inputs = self._replace_last(decoder_inputs, int_markers.EOS)[:, :-1]
        decoder_targets = target_turn[:, 1:]

        batch = {
                 'encoder_inputs': encoder_inputs,
                 'encoder_inputs_last_inds': self._get_last_inds(encoder_inputs, int_markers.PAD),
                 'decoder_inputs': decoder_inputs,
                 'decoder_inputs_last_inds': self._get_last_inds(decoder_inputs, int_markers.PAD),
                 'targets': decoder_targets,
                 'encoder_tokens': encode_tokens,
                 'decoder_tokens': decode_tokens,
                }
        return batch

    def _get_last_inds(self, inputs, stop_symbol):
        '''
        Return the last index which is not stop_symbol.
        inputs: (batch_size, input_size)
        '''
        assert type(stop_symbol) is int
        inds = np.argmax(inputs == stop_symbol, axis=1)
        inds[inds == 0] = inputs.shape[1]
        inds = inds - 1
        return inds

    def _get_token_turns(self, i, stage):
        if not hasattr(self.dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[stage][i] if i < len(dialogue.token_turns[stage]) else ''
                for dialogue in self.dialogues]

    def create_batches(self):
        self._normalize_dialogue()
        turn_batches = self._create_turn_batches()  # (batch_size, num_turns)
        # A sequence of batches should be processed in turn as the state of each batch is
        # passed on to the next batch
        batches = []
        enc, dec, tgt = Dialogue.ENC, Dialogue.DEC, Dialogue.TARGET
        for start_encode in (0, 1):
            encode_turn_ids = range(start_encode, self.num_turns-1, 2)
            batch_seq = [self._create_one_batch(turn_batches[enc][i], turn_batches[dec][i+1], turn_batches[tgt][i+1], self._get_token_turns(i, enc), self._get_token_turns(i+1, dec)) for i in encode_turn_ids]
            if start_encode == 1:
                # We still want to generate the first turn
                batch_seq.insert(0, self._create_one_batch(None, turn_batches[dec][0], turn_batches[tgt][0], None, self._get_token_turns(0, dec)))
            # Add agents and kbs
            agents = self._get_agent_batch(1 - start_encode)  # Decoding agent
            kbs = self._get_kb_batch(agents)
            batch = {
                     'agent': agents,
                     'kb': kbs,
                     'batch_seq': batch_seq,
                    }
            if self.use_kb:
                batch['graph'] = self._get_graph_batch(agents)
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
        if form == 'surface':
            return entity[0]
        elif form == 'type':
            return entity[1][1]
        elif form == 'canonical':
            return entity[1]
        elif form == 'graph':
            return entity[1]
        else:
            raise ValueError('Unknown entity form %s' % form)

    def process_utterance(self, utterance, stage):
        return [self.get_entity_form(x, self.entity_forms[stage]) if is_entity(x) else x for x in utterance]

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue.
        '''
        kbs = ex.scenario.kbs
        dialogue = Dialogue(kbs, ex.uuid)
        for e in ex.events:
            utterances = self.process_event(e, kbs[e.agent])
            if utterances:
                dialogue.add_utterance(e.agent, utterances)
        return dialogue

    def item_to_entities(self, item):
        '''
        Convert an item to a list of entities representing that item.
        '''
        entities = [(value, (value, type_)) for value, type_ in
            ((item[attr.name].lower(), self.attribute_types[attr.name])
                for attr in self.attributes)]
        return entities

    @classmethod
    def get_item_id(cls, kb, item):
        '''
        Return id of the item in kb.
        '''
        item_id = None
        for i, it in enumerate(kb.items):
            if it == item:
                item_id = i
                break
        if item_id is None:
            kb.dump()
            print item
        assert item_id is not None
        return item_id

    def process_event(self, e, kb):
        '''
        Convert event to two lists of tokens and entities for encoding and decoding.
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.entitylink(tokenize(e.data))
            if entity_tokens:
                return (entity_tokens, entity_tokens)
            else:
                return None
        elif e.action == 'select':
            # Convert an item to item-id (wrt to the speaker)
            item_id = self.get_item_id(kb, e.data)
            item_str = 'item-%d' % item_id
            # We use the entities to represent the item during encoding and item-id during decoding
            return ([markers.SELECT] + self.item_to_entities(e.data),
                    [markers.SELECT, (item_str, (item_str, 'item'))])
        else:
            raise ValueError('Unknown event action.')

    @classmethod
    def count_words(cls, examples):
        counts = defaultdict(int)
        for ex in examples:
            for event in ex.events:
                if event.action == 'message':
                    tokens = tokenize(event.data)
                    for token in tokens:
                        counts[token] += 1
        return counts

    def preprocess(self, examples):
        dialogues = []
        for ex in examples:
            d = self._process_example(ex)
            if len(d.agents) < 2:
                continue
                print 'Removing dialogue %s' % d.uuid
                for event in ex.events:
                    print event.to_dict()
            else:
                dialogues.append(d)
        return dialogues

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, preprocessor, schema, num_items, mappings=None, use_kb=False, copy=False):
        examples = {'train': train_examples or [], 'dev': dev_examples or [], 'test': test_examples or []}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.iteritems()}
        self.use_kb = use_kb  # Whether to generate graph
        self.copy = copy

        self.dialogues = {k: preprocessor.preprocess(v)  for k, v in examples.iteritems()}
        for fold, dialogues in self.dialogues.iteritems():
            print '%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold])

        if not mappings:
            mappings = create_mappings(self.dialogues['train'], schema, num_items, preprocessor.entity_forms.values())
        self.mappings = mappings

        self.textint_map = TextIntMap(mappings['vocab'], mappings['entity'], preprocessor)
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
        dialogues.sort(key=lambda d: len(d.turns))
        N = len(dialogues)
        dialogue_batches = []
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.extend(DialogueBatch(dialogue_batch, self.use_kb).create_batches())
            start = end
        return dialogue_batches

    def create_graph(self, dialogues):
        if not self.use_kb:
            return
        for dialogue in dialogues:
            dialogue.create_graph()

    def reset_graph(self, dialogue_batches):
        if not self.use_kb:
            return
        for dialogue_batch in dialogue_batches:
            for graph in dialogue_batch['graph'].graphs:
                graph.reset()

    def generator(self, name, batch_size, shuffle=True):
        dialogues = self.dialogues[name]
        for dialogue in dialogues:
            dialogue.convert_to_int()
        # NOTE: we assume that GraphMetadata has been constructed before DataGenerator is called
        self.create_graph(dialogues)
        dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)
        yield len(dialogue_batches)
        inds = range(len(dialogue_batches))
        while True:
            if shuffle:
                random.shuffle(inds)
            for ind in inds:
                yield dialogue_batches[ind]
            # We want graphs clean of dialgue history for the new epoch
            self.reset_graph(dialogue_batches)
