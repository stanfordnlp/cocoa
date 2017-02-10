__author__ = 'anushabala'
from session import Session
from src.model.graph import Graph, GraphBatch
from src.model.preprocess import markers, word_to_num
from src.model.vocab import is_entity, Vocabulary
from src.model.evaluate import pred_to_token
import numpy as np
import random
import re
from itertools import izip

num_to_word = {v: k for k, v in word_to_num.iteritems()}

class NeuralSession(Session):
    """
    NeuralSession represents a dialogue agent backed by a neural model. This class stores the knowledge graph,
    tensorflow session, and other model-specific information needed to update the model (when a message is received)
    or generate a response using the model (when send() is called).
    This class is closely related to but different from NeuralSystem. NeuralSystem is the class that actually loads the
    model from disk, while NeuralSession represents a specific instantiation of that model, and contains information
    (like the agent index and the knowledge base) that are specific to an agent in the current dialogue.
    """
    def __init__(self, agent, kb, env):
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.model = env.model
        self.kb = kb
        self.matched_item = None
        self.sent_entity = False
        self.mentioned_entities = set()
        #self.log = open('chat.debug.log', 'a')
        #self.log.write('-------------------------------------\n')

        self.capitalize = random.choice([True, False])
        self.numerical = random.choice([True, False])

    def encode(self, entity_tokens):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def receive(self, event):
        #self.log.write('receive event:%s\n' % str(event.to_dict()))
        # Reset status
        self.sent_entity = False
        # Parse utterance
        if event.action == 'select':
            self.matched_item = self._match(event.data)
            if self.matched_item is None:
                entity_tokens = [markers.SELECT] + self.env.preprocessor.item_to_entities(event.data, self.kb.attributes)
            else:
                # Got a match; we're done.
                return
        elif event.action == 'message':
            entity_tokens = self.env.preprocessor.process_event(event, self.kb, mentioned_entities=self.mentioned_entities, known_kb=False)
            print entity_tokens[0]
            # Empty message
            if entity_tokens is None:
                return
            else:
                # Take the encoding version of sequence
                entity_tokens = entity_tokens[0]
        else:
            raise ValueError('Unknown event action %s.' % event.action)
        for token in entity_tokens:
            if is_entity(token):
                self.mentioned_entities.add(token[1][0])
        entity_tokens += [markers.EOS]

        self.encode(entity_tokens)

    def _has_entity(self, tokens):
        for token in tokens:
            if is_entity(token):
                return True
        return False

    def naturalize(self, tokens):
        '''
        Process the tokens to add variation, e.g. capitalization, number representation.
        '''
        # Map wrong numerics to word, e.g. not that 1
        for i, (w1, w2) in enumerate(izip(tokens, tokens[1:])):
            if w1 in ('this', 'that', 'the') and w2 == '1':
                tokens[i+1] == 'one'
        if self.capitalize:
            tokens[0] = tokens[0].title()
            tokens = ['I' if x == 'i' else x for x in tokens]
        # Model output is numerical by default
        if not self.numerical:
            tokens = [num_to_word[x] if x in num_to_word else x for x in tokens]
        return tokens

    def attach_punct(self, s):
        s = re.sub(r' ([.,!?;])', r'\1', s)
        s = re.sub(r'\.{3,}', r'...', s)
        return s

    def send(self):
        # Don't send consecutive utterances with entities
        if self.sent_entity and not self.env.consecutive_entity:
            return None
        if self.matched_item is not None:
            return self.select(self.matched_item)
        for i in xrange(1):
            tokens = self.decode()
            if tokens is not None:
                break
        if tokens is None:
            return None
        if self._has_entity(tokens):
            self.sent_entity = True
        else:
            self.sent_entity = False
        for token in tokens:
            if is_entity(token):
                self.mentioned_entities.add(token[1][0])
        if self.env.realizer is None:
            tokens = [x if not is_entity(x) else x[0] for x in tokens]
        else:
            tokens = self.env.realizer.realize_entity(tokens)
        if len(tokens) > 1 and tokens[0] == markers.SELECT and tokens[1].startswith('item-'):
            item_id = int(tokens[1].split('-')[1])
            self.selected_items.add(item_id)
            item = self.kb.items[item_id]
            return self.select(item)
        tokens = self.naturalize(tokens)
        s = self.attach_punct(' '.join(tokens))
        return self.message(s)

class RNNNeuralSession(NeuralSession):
    '''
    RNN Session use the vanila seq2seq model without graphs.
    '''
    def __init__(self, agent, kb, env):
        super(RNNNeuralSession, self).__init__(agent, kb, env)
        self.encoder_state = None
        self.decoder_state = None
        self.encoder_output_dict = None

        self.new_turn = False
        self.end_turn = False

        self.sent_entity = False
        self.selected_items = set()

    @classmethod
    def _get_last_inds(cls, inputs):
        '''
        For batch_size=1, just return the last index.
        '''
        return np.ones(inputs.shape[0], dtype=np.int32) * (inputs.shape[1] - 1)

    def _process_entity_tokens(self, entity_tokens, stage):
        int_inputs = np.reshape(self.env.textint_map.text_to_int(entity_tokens), [1, -1])
        inputs, entities = self.env.textint_map.process_entity(int_inputs, stage)
        return inputs, entities

    def _encoder_args(self, entity_tokens):
        #inputs = np.reshape(self.env.textint_map.text_to_int(entity_tokens, 'encoding'), [1, -1])
        inputs, entities = self._process_entity_tokens(entity_tokens, 'encoding')
        #self.log.write('encoder entities:%s\n' % str(entities))
        encoder_args = {'inputs': inputs,
                'last_inds': self._get_last_inds(inputs),
                'init_state': self.encoder_state,
                'entities': entities,
                }
        return encoder_args

    def _decoder_init_state(self, sess):
        return self.encoder_state

    def _decoder_args(self, init_state, inputs):
        decoder_args = {'inputs': inputs,
                'last_inds': np.zeros([1], dtype=np.int32),
                'init_state': init_state,
                'textint_map': self.env.textint_map,
                }
        return decoder_args

    def _update_states(self, sess, decoder_output_dict, entity_tokens):
        self.decoder_state = decoder_output_dict['final_state']
        self.encoder_state = decoder_output_dict['final_state']

    def decode(self):
        sess = self.env.tf_session

        if self.encoder_output_dict is None:
            self.encode([markers.EOS])

        if self.new_turn:
            # If this is a new turn, we need to continue from the encoder outputs, thus need
            # to compute init_state (including attention and context). GO indicates a new turn
            # as opposed to EOS within a turn.
            start_symbol = markers.GO
            init_state = self._decoder_init_state(sess)
        else:
            assert self.decoder_state is not None
            init_state = self.decoder_state
            start_symbol = markers.EOS

        inputs = np.reshape(self.env.textint_map.text_to_int([start_symbol], 'decoding'), [1, 1])

        decoder_args = self._decoder_args(init_state, inputs)
        decoder_output_dict = self.model.decoder.decode(sess, self.env.max_len, batch_size=1, stop_symbol=self.env.stop_symbol, **decoder_args)

        entity_tokens = self._pred_to_token(decoder_output_dict['preds'])[0]
        if not self._is_valid(entity_tokens):
            return None
        #self.log.write('decode:%s\n' % str(entity_tokens))
        self._update_states(sess, decoder_output_dict, entity_tokens)

        # Text message
        if self.new_turn:
            self.new_turn = False
        return entity_tokens
        #return [x if not is_entity(x) else x[0] for x in entity_tokens]

    def _is_valid(self, tokens):
        if not tokens:
            return False
        if Vocabulary.UNK in tokens:
            return False
        if tokens[0] == markers.SELECT:
            if len(tokens) > 1 and isinstance(tokens[1], tuple) and tokens[1][0].startswith('item-'):
                item_id = int(tokens[1][0].split('-')[1])
                if item_id in self.selected_items or item_id >= len(self.kb.items):
                    return False
                else:
                    return True
            else:
                return False
        else:
            for token in tokens:
                if isinstance(token, tuple) and token[0].startswith('item-'):
                    return False
        return True

    def encode(self, entity_tokens):
        encoder_args = self._encoder_args(entity_tokens)
        #self.log.write('encode:%s\n' % str(entity_tokens))
        self.encoder_output_dict = self.model.encoder.encode(self.env.tf_session, **encoder_args)
        self.encoder_state = self.encoder_output_dict['final_state']
        self.new_turn = True

    def _pred_to_token(self, preds):
        if self.env.copy:
            preds = self.graph.copy_preds(preds, self.env.vocab.size)
        entity_tokens, _ = pred_to_token(preds, self.env.stop_symbol, self.env.remove_symbols, self.env.textint_map)
        entity_tokens = [[(x[0], x) if is_entity(x) else x for x in toks] for toks in entity_tokens]
        return entity_tokens

    def _match(self, item):
        for it in self.kb.items:
            if it == item:
                return it
        return None

class GraphNeuralSession(RNNNeuralSession):
    def __init__(self, agent, kb, env):
        super(GraphNeuralSession, self).__init__(agent, kb, env)
        self.graph = GraphBatch([Graph(kb)])

        self.utterances = None
        self.context = None
        self.graph_data = None
        self.init_checklists = None

    def encode(self, entity_tokens):
        super(GraphNeuralSession, self).encode(entity_tokens)
        self.context = self.encoder_output_dict['context']

    def _encoder_args(self, entity_tokens):
        encoder_args = super(GraphNeuralSession, self)._encoder_args(entity_tokens)
        graph_data = self.graph.get_batch_data([entity_tokens], None, encoder_args['entities'], None, self.utterances, self.env.vocab)
        encoder_args['update_entities'] = graph_data['encoder_entities']
        #self.log.write('encoder update entities:%s\n' % str(encoder_args['update_entities']))
        encoder_args['entities'] = graph_data['encoder_nodes']
        encoder_args['utterances'] = graph_data['utterances']
        encoder_args['graph_data'] = graph_data
        return encoder_args

    def _decoder_init_state(self, sess):
        self.init_checklists = self.graph.get_zero_checklists(1)
        init_state = self.model.decoder.compute_init_state(sess,
                self.encoder_state,
                self.encoder_output_dict['final_output'],
                self.context,
                self.init_checklists)
        return init_state

    def _decoder_args(self, init_state, inputs):
        decoder_args = super(GraphNeuralSession, self)._decoder_args(init_state, inputs)
        decoder_args['init_checklists'] = self.init_checklists
        decoder_args['entities'] = self.graph.get_zero_entities(1)
        decoder_args['graphs'] = self.graph
        decoder_args['vocab'] = self.env.vocab
        return decoder_args

    def _update_states(self, sess, decoder_output_dict, entity_tokens):
        self.decoder_state = decoder_output_dict['final_state']
        self.encoder_state = decoder_output_dict['final_state'][0]

        # Update graph and utterances
        graph_data = self.graph.get_batch_data(None, [entity_tokens], None, None, self.utterances, self.env.vocab)

        #self.log.write('decoder update entities:%s\n' % str(graph_data['decoder_entities']))
        self.utterances, self.context = self.model.decoder.update_context(sess, graph_data['decoder_entities'], decoder_output_dict['final_output'], decoder_output_dict['utterance_embedding'], graph_data['utterances'], graph_data)
        self.init_checklists = decoder_output_dict['checklists']
