'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from vocab import Vocabulary, is_entity
from graph import Graph
from itertools import chain

# Special symbols
EOT = '</t>'  # End of turn
EOU = '</s>'  # End of utterance
SELECT = '<select>'
markers = (EOT, EOU, SELECT)

def tokenize(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    # Split on punctuation
    tokens = re.findall(r"[\w']+|[.,!?;]", utterance)
    return tokens

def build_schema_mappings(offset, schema):
    entity_map = Vocabulary(offset=offset, unk=True, pad=False)
    for type_, values in schema.values.iteritems():
        entity_map.add_words(((value.lower(), type_) for value in values))
    offset += entity_map.size

    relation_map = Vocabulary(offset=offset, unk=False, pad=False)
    attribute_types =  schema.get_attributes()  # {attribute_name: value_type}
    relation_map.add_words((a.lower() for a in attribute_types.keys()))

    return entity_map, relation_map

def build_vocab(dialogues, special_symbols=[], add_entity=True):
    vocab = Vocabulary(offset=0, unk=True, pad=True)

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
        for turn in dialogue.turns:
            for token in chain.from_iterable(turn):
                _add_token(token)

    # Add special symbols
    vocab.add_words(special_symbols)
    return vocab

class Dialogue(object):
    def __init__(self, kbs, turns=None, agents=None):
        self.kbs = kbs
        self.turns = turns or []
        self.agents = agents or []
        assert len(self.turns) == len(self.agents)
        self.is_int = False  # Whether we've converted it to integers
        self.flattened = False

    def add_utterance(self, agent, utterance):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            self.turns[-1].append(utterance)
        else:
            self.agents.append(agent)
            self.turns.append([utterance])

    def _text_to_int(self, tokens, vocab, entity_map):
        # Use canonical string for entities
        return [vocab.to_ind(token) if not is_entity(token) else entity_map.to_ind(token[1]) if entity_map else vocab.to_ind(token[1]) for token in tokens]

    def convert_to_int(self, vocab, entity_map=None):
        for i, turn in enumerate(self.turns):
           self.turns[i] = [self._text_to_int(utterance, vocab, entity_map) for utterance in turn]
        self.is_int = True

    def flatten_turns(self):
        '''
        Flatten turns to a list of tokens with </s> between utterances and </t> at the end.
        '''
        if self.flattened:
            return
        turns = self.turns
        for i, turn in enumerate(turns):
            for utterance in turn:
                utterance.append(EOU)
            turns[i] = [x for x in chain.from_iterable(turn)] + [EOT]
        self.flattened = True

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        turns, agents = self.turns, self.agents
        for i in xrange(len(turns), num_turns):
            agents.append(1 - agents[-1])
            turns.append([])

class DialogueBatch(object):
    def __init__(self, dialogues):
        self.dialogues = dialogues

    def _normalize_dialogue(self):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([len(d.turns) for d in self.dialogues])
        for dialogue in self.dialogues:
            dialogue.flatten_turns()
            dialogue.pad_turns(max_num_turns)
        self.num_turns = len(self.dialogues[0].turns)

    def _normalize_turn(self, turn_batch):
        '''
        All turns at the same time step should have the same number of tokens.
        '''
        max_num_tokens = max([len(t) for t in turn_batch])
        batch_size = len(turn_batch)
        T = np.full([batch_size, max_num_tokens], PAD, dtype=np.int32)
        for i, turn in enumerate(turn_batch):
            T[i, :len(turn)] = turn
        return T

    def _create_turn_batches(self):
        turn_batches = [self._normalize_turn(
                [dialogue.turns[i] for dialogue in self.dialogues])
                for i in xrange(self.num_turns)]
        return turn_batches

    def _get_agent_batch(self, i):
        return [dialogue.agents[i] for dialogue in self.dialogues]

    def _get_kb_batch(self, i):
        return [dialogue.kbs[dialogue.agents[i]] for dialogue in self.dialogues]

    def _create_one_batch(self, encode_turn, decode_turn):
        decoder_go = np.full([decode_turn.shape[0], 1], EOT, dtype=np.int32)
        # NOTE: decoder_go is inserted to padded turn as well, i.e. </t> <pad> ...
        # This probably doesn't matter
        decoder_inputs = np.concatenate((decoder_go, decode_turn[:, :-1]), axis=1)
        decoder_targets = decode_turn
        batch = {
                 'encoder_inputs': encode_turn,
                 'decoder_inputs': decoder_inputs,
                 'decoder_targets': decoder_targets
                }
        return batch

    def create_batches(self):
        self._normalize_dialogue()
        turn_batches = self._create_turn_batches()  # (batch_size, num_turns)
        # A sequence of batches should be processed in turn as the state of each batch is
        # passed on to the next batch
        batches = []
        for start_encode in (0, 1):
            encode_turn_ids = range(start_encode, self.num_turns-1, 2)
            batch_seq = [self._create_one_batch(turn_batches[i], turn_batches[i+1]) for i in encode_turn_ids]
            if start_encode == 1:
                # We still want to generate the first turn
                batch_seq.insert(0, self._create_one_batch(None, turn_batches[0]))
            # Add agents and kbs
            batch = {
                     'agent': self._get_agent_batch(start_encode + 1),
                     'kb': self._get_kb_batch(start_encode + 1),
                     'batch_seq': batch_seq,
                    }
            batches.append(batch)
        return batches

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, schema, lexicon, mappings=None, use_kb=False):
        # TODO: change basic/dataset so that it uses the dict structure
        examples = {'train': train_examples or [], 'dev': dev_examples or [], 'test': test_examples or []}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.iteritems()}
        self.lexicon = lexicon
        self.use_kb = use_kb

        self.dialogues = self.preprocess(examples)

        if not mappings:
            self.create_mappings(schema)
        else:
            self.vocab = mappings['vocab']
            self.entity_map = mappings['entity']
            self.relation_map = mappings['relation']
        print 'Vocabulary size:', self.vocab.size
        print 'Entity size:', self.entity_map.size
        print 'Relation size:', self.relation_map.size

        self.convert_to_int()

    def create_mappings(self, schema):
        # NOTE: vocab, entity_map and relation_map have disjoint mappings so that we can
        # tell if an integer represents an entity or a token
        self.vocab = build_vocab(self.dialogues['train'], markers, add_entity=(not self.use_kb))
        self.entity_map, self.relation_map = build_schema_mappings(self.vocab.size, schema)

    def convert_to_int(self):
        '''
        Conver tokens to integers.
        '''
        entity_map = self.entity_map if self.use_kb else None
        for _, dialogues in self.dialogues.iteritems():
            for dialogue in dialogues:
                dialogue.convert_to_int(self.vocab, entity_map=entity_map)
        global PAD, EOT, EOU
        EOT = self.vocab.to_ind(EOT)
        EOU = self.vocab.to_ind(EOU)
        PAD = self.vocab.to_ind(self.vocab.PAD)

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue.
        '''
        kbs = ex.scenario.kbs
        dialogue = Dialogue(kbs)
        for e in ex.events:
            utterance = self._process_event(e, kbs[e.agent])
            dialogue.add_utterance(e.agent, utterance)
        return dialogue

    def _process_event(self, e, kb):
        '''
        Convert event to a list of tokens and entities.
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.entitylink(tokenize(e.data))
        elif e.action == 'select':
            # Check which item is selected
            item_id = None
            for i, item in enumerate(kb.items):
                if item == e.data:
                    item_id = i
                    break
            assert item_id is not None
            item_str = 'item-%d' % item_id
            # Treat item as an entity
            entity_tokens = [SELECT, (item_str, (item_str, 'item'))]
        else:
            raise ValueError('Unknown event action.')
        return entity_tokens

    def preprocess(self, dataset):
        dialogues = {}
        for fold, examples in dataset.iteritems():
            dialogues[fold] = [self._process_example(ex) for ex in examples]
        return dialogues

    def create_dialogue_batches(self, dialogues, batch_size):
        dialogues.sort(key=lambda d: len(d.turns))
        N = len(dialogues)
        dialogue_batches = []
        start = 0
        while start < N:
            end = start + batch_size
            # We don't have enough examples for the last batch; repeat examples in the
            # previous batch. TODO: repeated examples should have lower weights.
            if end > N:
                dialogue_batch = dialogues[-batch_size:]
            else:
                dialogue_batch = dialogues[start:end]
            dialogue_batches.extend(DialogueBatch(dialogue_batch).create_batches())
            start = end
        return dialogue_batches

    def train_generator(self, name, batch_size, shuffle=True):
        dialogue_batches = self.create_dialogue_batches(self.dialogues[name], batch_size)
        inds = range(len(dialogue_batches))
        while True:
            if shuffle:
                random.shuffle(inds)
            for ind in inds:
                yield dialogue_batches[ind]
