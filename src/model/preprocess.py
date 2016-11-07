'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from src.model.vocab import Vocabulary, is_entity
from src.model.graph import Graph, GraphBatch, inv_rel
from itertools import chain, izip
from collections import namedtuple
import copy

SpecialSymbols = namedtuple('SpecialSymbols', ['EOT', 'EOS', 'GO', 'SELECT', 'PAD'])
markers = SpecialSymbols(EOT='</t>', EOS='</s>', GO='<go>', SELECT='<select>', PAD='<pad>')

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

def build_vocab(dialogues, special_symbols=[], add_entity=True):
    vocab = Vocabulary(offset=0, unk=True)

    def _add_token(token):
        # Surface form of the entity
        if is_entity(token):
            vocab.add_word(token[0])
            if add_entity:
                vocab.add_word(token[1])
        else:
            vocab.add_word(token)

    # Add words
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for turns in dialogue.turns:
            for turn in turns:
                for token in chain.from_iterable(turn):
                    _add_token(token)

    # Add special symbols
    vocab.add_words(special_symbols)
    return vocab

def create_mappings(dialogues, schema, num_items):
    vocab = build_vocab(dialogues, markers, add_entity=True)
    entity_map, relation_map = build_schema_mappings(schema, num_items)
    return {'vocab': vocab,
            'entity': entity_map,
            'relation': relation_map,
            }

def entity_to_vocab(inputs, vocab):
    '''
    Convert entity ids to vocab ids. In preprocessing we have replaced all entities to
    entity ids (offset by vocab.size). Now to process them during encoding we need to
    map them back to vocab ids. Note that some entities will become UNK.
    TODO: better entity encoding.
    '''
    new_inputs = np.array(inputs)
    for i, array in enumerate(new_inputs):
        for j, value in enumerate(array):
            if value >= vocab.size:
                entity = Graph.metadata.entity_map.to_word(value - vocab.size)
                new_inputs[i][j] = vocab.to_ind(entity)
    return new_inputs

class TextIntMap(object):
    '''
    Map between text and int for visualizing results.
    '''
    def __init__(self, vocab, entity_map, copy):
        self.vocab = vocab
        self.entity_map = entity_map
        self.copy = copy

    def text_to_int(self, tokens):
        '''
        Look up tokens in vocab; if entity_map is not used, look up for an entity using its
        canonical form (token[1]) in vocab; otherwise look up it in the entity_map and make
        sure the id is disjoint with the vocab id by adding an offset.
        '''
        if not self.copy:
            return [self.vocab.to_ind(token) if not is_entity(token) else self.vocab.to_ind(token[1]) for token in tokens]
        else:
            offset = self.vocab.size
            return [self.vocab.to_ind(token) if not is_entity(token) else self.entity_map.to_ind(token[1]) + offset for token in tokens]

    def int_to_text(self, inds):
        '''
        Inverse of text_to_int.
        '''
        if not self.copy:
            return [self.vocab.to_word(ind) for ind in inds]
        else:
            return [self.vocab.to_word(ind) if ind < self.vocab.size else self.entity_map.to_word(ind - self.vocab.size) for ind in inds]

class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1

    def __init__(self, kbs, uuid):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.kbs = kbs
        # turns[0] and turns[1] are  utterances from the encoder's and the decoder's perspectives
        # turns: input tokens of encoder and decoder, later converted to integers
        self.turns = ([], [])
        self.agents = []
        self.is_int = False  # Whether we've converted it to integers
        self.flattened = False

    def create_graph(self):
        assert not hasattr(self, 'graphs')
        self.graphs = [Graph(kb) for kb in self.kbs]

    def _process_encoder_utterance(self, utterance):
        '''
        In process_event, we represent a select action as [SELECT, item id, item entities].
        For encoding, we want to use [SELECT, item entities].
        '''
        if utterance[0] == markers.SELECT:
            return utterance[:1] + utterance[2:]
        return utterance

    def _process_decoder_utterance(self, utterance):
        '''
        In process_event, we represent a select action as [SELECT, item id, item entities].
        For decoding, we want to use [SELECT, item id].
        '''
        if utterance[0] == markers.SELECT:
            return utterance[:2]
        return utterance

    def _process_utterance(self, utterance, stage):
        if stage == self.ENC:
            return self._process_encoder_utterance(utterance)
        elif stage == self.DEC:
            return self._process_decoder_utterance(utterance)
        else:
            raise ValueError

    def add_utterance(self, agent, utterance):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            for i in xrange(2):
                self.turns[i][-1].append(self._process_utterance(utterance, i))
        else:
            self.agents.append(agent)
            for i in xrange(2):
                self.turns[i].append([self._process_utterance(utterance, i)])

    def convert_to_int(self, keep_tokens=False):
        if self.is_int:
            return
        if keep_tokens:
            self.token_turns = copy.deepcopy(self.turns)
        for i, turns in enumerate(self.turns):
            for j, turn in enumerate(turns):
                self.turns[i][j] = [self.textint_map.text_to_int(utterance) for utterance in turn]
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
                self.turns[i][j] = [x for x in chain.from_iterable(turn)] + [int_markers.EOT]

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
        assert len(turns[0]) == len(turns[1])
        for i in xrange(len(turns[0]), num_turns):
            agents.append(1 - agents[-1])
            turns[0].append([])
            turns[1].append([])

class DialogueBatch(object):
    def __init__(self, dialogues, use_kb=False):
        self.dialogues = dialogues
        self.use_kb = use_kb

    def _normalize_dialogue(self):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        # len(turns[0]) == len(turns[1])
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
        for i in xrange(2):
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

    def _create_one_batch(self, encode_turn, decode_turn, encode_tokens, decode_tokens):
        if encode_turn is not None:
            # Remove <go> and </t> at the beginning and end of utterance
            encoder_inputs = np.copy(encode_turn[:, 1:-1])
            encoder_inputs[encoder_inputs == int_markers.EOT] = int_markers.PAD
        else:
            batch_size = decode_turn.shape[0]
            # If there's no input to encode, use </s> as the encoder input.
            encoder_inputs = np.full((batch_size, 1), int_markers.EOS, dtype=np.int32)
            # encode_tokens are empty lists
            encode_tokens = [[''] for _ in xrange(batch_size)]

        # Decoder inptus: start from <go> to generate, i.e. <go> <token>
        decoder_inputs = decode_turn[:, :-1]
        decoder_targets = decode_turn[:, 1:]

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
        enc, dec = Dialogue.ENC, Dialogue.DEC
        for start_encode in (0, 1):
            encode_turn_ids = range(start_encode, self.num_turns-1, 2)
            batch_seq = [self._create_one_batch(turn_batches[enc][i], turn_batches[dec][i+1], self._get_token_turns(i, enc), self._get_token_turns(i+1, dec)) for i in encode_turn_ids]
            if start_encode == 1:
                # We still want to generate the first turn
                batch_seq.insert(0, self._create_one_batch(None, turn_batches[dec][0], None, self._get_token_turns(0, dec)))
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
    def __init__(self, schema, lexicon):
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue.
        '''
        kbs = ex.scenario.kbs
        dialogue = Dialogue(kbs, ex.uuid)
        for e in ex.events:
            utterance = self.process_event(e, kbs[e.agent])
            if utterance:
                dialogue.add_utterance(e.agent, utterance)
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
        Convert event to a list of tokens and entities.
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.entitylink(tokenize(e.data))
        elif e.action == 'select':
            item_id = self.get_item_id(kb, e.data)
            item_str = 'item-%d' % item_id
            # Convert an item to item-id (wrt to the speaker) and a list of entities (wrt to the listner)
            # We use the entities to represent the item during encoding and item-id during decoding
            entity_tokens = [markers.SELECT, (item_str, (item_str, 'item'))] + self.item_to_entities(e.data)
        else:
            raise ValueError('Unknown event action.')
        return entity_tokens

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
            mappings = create_mappings(self.dialogues['train'], schema, num_items)
        self.mappings = mappings

        self.textint_map = TextIntMap(mappings['vocab'], mappings['entity'], copy)
        Dialogue.textint_map = self.textint_map

        global int_markers
        int_markers = SpecialSymbols(*[mappings['vocab'].to_ind(m) for m in markers])

    def convert_to_int(self):
        '''
        Convert tokens to integers.
        '''
        for fold, dialogues in self.dialogues.iteritems():
            for dialogue in dialogues:
                dialogue.convert_to_int(keep_tokens=True)

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
            dialogue.convert_to_int(keep_tokens=True)
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
