'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import numpy as np
from vocab import Vocabulary, is_entity
from graph import Graph, GraphBatch
from itertools import chain, izip
import copy

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

def build_schema_mappings(schema, offset=0, disjoint=False):
    offset = offset if disjoint else 0
    entity_map = Vocabulary(offset=offset, unk=True, pad=False)
    for type_, values in schema.values.iteritems():
        entity_map.add_words(((value.lower(), type_) for value in values))

    offset = offset + entity_map.size if disjoint else 0
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

#def text_to_int(tokens, vocab, entity_map):
#    # Use canonical string for entities
#    return [vocab.to_ind(token) if not is_entity(token) else entity_map.to_ind(token[1]) if entity_map else vocab.to_ind(token[1]) for token in tokens]

def text_to_int(tokens, vocab):
    # We look for the caninical form (token[1]) of an entity in the vocab
    return [vocab.to_ind(token) if not is_entity(token) else vocab.to_ind(token[1]) for token in tokens]

def int_to_text(inds, vocab):
    return [vocab.to_word(ind) for ind in inds]

class Dialogue(object):
    def __init__(self, kbs, turns=None, agents=None):
        '''
        Dialogue data that is needed by the model.
        '''
        self.kbs = kbs
        self.turns = turns or []
        self.agents = agents or []
        assert len(self.turns) == len(self.agents)
        self.is_int = False  # Whether we've converted it to integers
        self.flattened = False

    def create_graph(self):
        self.graphs = [Graph(kb) for kb in self.kbs]

    def add_utterance(self, agent, utterance):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            self.turns[-1].append(utterance)
        else:
            self.agents.append(agent)
            self.turns.append([utterance])

    def convert_to_int(self, vocab, entity_map=None, keep_tokens=False):
        if self.is_int:
            return
        if keep_tokens:
            self.token_turns = copy.copy(self.turns)
        for i, turn in enumerate(self.turns):
            self.turns[i] = [text_to_int(utterance, vocab) for utterance in turn]
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

        if hasattr(self, 'token_turns'):
            turns = self.token_turns
            for i, turn in enumerate(turns):
                turns[i] = [x for x in chain.from_iterable(turn)]

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
    def __init__(self, dialogues, use_kb=False):
        self.dialogues = dialogues
        self.use_kb = use_kb

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
        T = np.full([batch_size, max_num_tokens+1], PAD, dtype=np.int32)
        # Insert </t> at the beginning at each turn because for decoding we want to
        # start from </t> to generate
        T[:, 0] = EOT
        for i, turn in enumerate(turn_batch):
            T[i, 1:len(turn)+1] = turn
        return T

    def _create_turn_batches(self):
        turn_batches = [self._normalize_turn(
                [dialogue.turns[i] for dialogue in self.dialogues])
                for i in xrange(self.num_turns)]
        return turn_batches

    def _get_agent_batch(self, i):
        return [dialogue.agents[i] for dialogue in self.dialogues]

    def _get_kb_batch(self, agents):
        return [dialogue.kbs[agent] for dialogue, agent in izip(self.dialogues, agents)]

    def _get_graph_batch(self, agents):
        return GraphBatch([dialogue.graphs[agent] for dialogue, agent in izip(self.dialogues, agents)])

    def _create_one_batch(self, encode_turn, decode_turn, encode_tokens, decode_tokens):
        # Encoder inptus: remove </t> since it's used to start decoding, i.e. token <pad>
        if encode_turn is not None:
            # </t> at the beginning and end of utterance
            # Replace the end </t> but keep the beginning one to separate it from the previous utterance
            encoder_inputs = np.copy(encode_turn[:, :-1])
            encoder_inputs[encoder_inputs == EOT] = PAD
            encoder_inputs[:, 0] = EOT
        else:
            # If there's no input to encode, use </t> as the encoder input.
            encoder_inputs = decode_turn[:, [0]]
        # Decoder inptus: start from </t> to generate, i.e. </t> <token> NOTE: </t> is
        # inserted to padded turns as well, i.e. </t> <pad>, probably doesn't matter though..
        decoder_inputs = decode_turn[:, :-1]
        decoder_targets = decode_turn[:, 1:]
        batch = {
                 'encoder_inputs': encoder_inputs,
                 'encoder_inputs_last_inds': self._get_last_inds(encoder_inputs, PAD),
                 'decoder_inputs': decoder_inputs,
                 'decoder_inputs_last_inds': self._get_last_inds(decoder_inputs, PAD),
                 'targets': decoder_targets
                }
        if encode_tokens is not None or decode_tokens is not None:
            batch['encoder_tokens'] = encode_tokens
            batch['decoder_tokens'] = decode_tokens
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

    def _get_token_turns(self, i):
        if not hasattr(self.dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[i] if i < len(dialogue.token_turns) else None
                for dialogue in self.dialogues]

    def create_batches(self):
        self._normalize_dialogue()
        turn_batches = self._create_turn_batches()  # (batch_size, num_turns)
        # A sequence of batches should be processed in turn as the state of each batch is
        # passed on to the next batch
        batches = []
        for start_encode in (0, 1):
            encode_turn_ids = range(start_encode, self.num_turns-1, 2)
            batch_seq = [self._create_one_batch(turn_batches[i], turn_batches[i+1], self._get_token_turns(i), self._get_token_turns(i+1)) for i in encode_turn_ids]
            if start_encode == 1:
                # We still want to generate the first turn
                batch_seq.insert(0, self._create_one_batch(None, turn_batches[0], None, self._get_token_turns(0)))
            # Add agents and kbs
            agents = self._get_agent_batch(start_encode + 1)  # Decoding agent
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

# TODO: separate Preprocessor (currently the constructor of DataGenerator)
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

        # Convert dialogue utterances to integers
        self.convert_to_int()

        # Convert special symbols to integer
        global PAD, EOT, EOU
        EOT = self.vocab.to_ind(EOT)
        EOU = self.vocab.to_ind(EOU)
        PAD = self.vocab.to_ind(self.vocab.PAD)

    def create_mappings(self, schema):
        # NOTE: DISABLED NOW.
        # vocab, entity_map and relation_map have disjoint mappings so that we can
        # tell if an integer represents an entity or a token
        self.vocab = build_vocab(self.dialogues['train'], markers, add_entity=True)
        self.entity_map, self.relation_map = build_schema_mappings(schema, offset=self.vocab.size, disjoint=False)

    def convert_to_int(self, entity_map=None):
        '''
        Convert tokens to integers.
        '''
        for fold, dialogues in self.dialogues.iteritems():
            #keep_tokens = True if fold in ['dev', 'test'] else False
            keep_tokens = True
            for dialogue in dialogues:
                dialogue.convert_to_int(self.vocab, entity_map=entity_map, keep_tokens=keep_tokens)

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
            # previous batch. TODO: repeated examples should have lower weights, or better,
            # make batch_size a variable
            if end > N:
                # dialogue_batch = dialogues[-batch_size:]
                # We need deepcopy here because data in Graph will be written
                dialogue_batch = dialogues[start:] + [copy.deepcopy(dialogue) for dialogue in dialogues[start-1:start-(end-N+1):-1]]
                assert len(dialogue_batch) == batch_size
            else:
                dialogue_batch = dialogues[start:end]
            dialogue_batches.extend(DialogueBatch(dialogue_batch, self.use_kb).create_batches())
            start = end
        return dialogue_batches

    def create_graph(self, dialogues):
        if not self.use_kb:
            return
        for dialogue in dialogues:
            dialogue.create_graph()

    def reset_graph(self, dialogues):
        if not self.use_kb:
            return
        for dialogue in dialogues:
            for graph in dialogue.graphs:
                graph.reset()

    def generator(self, name, batch_size, shuffle=True):
        dialogues = self.dialogues[name]
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
            self.reset_graph(dialogues)
