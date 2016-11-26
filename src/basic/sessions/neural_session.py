__author__ = 'anushabala'
from session import Session
from src.model.graph import Graph, GraphBatch
from src.model.preprocess import markers
from src.model.vocab import is_entity
from src.model.evaluate import pred_to_token
import numpy as np
import random

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

    def encode(self, entity_tokens):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def receive(self, event):
        # Parse utterance
        if event.action == 'select':
            self.matched_item = self._match(event.data)
            if self.matched_item is None:
                entity_tokens = [markers.SELECT] + self.env.preprocessor.item_to_entities(event.data)
            else:
                # Got a match; we're done.
                return
        elif event.action == 'message':
            entity_tokens = self.env.preprocessor.process_event(event, self.kb)
            # Empty message
            if entity_tokens is None:
                return
            else:
                # Take the encoding version of sequence
                entity_tokens = entity_tokens[0]
        else:
            raise ValueError('Unknown event action.')
        entity_tokens += [markers.EOS]

        self.encode(entity_tokens)

    def send(self):
        if self.matched_item is not None:
            return self.select(self.matched_item)
        #if random.random() < 0.5:  # Wait randomly
        #    return None
        tokens = self.decode()
        if len(tokens) > 1 and tokens[0] == markers.SELECT and tokens[1].startswith('item-'):
            item = self.kb.items[int(tokens[1].split('-')[1])]
            return self.select(item)
        return self.message(' '.join(tokens))

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

    @classmethod
    def _get_last_inds(cls, inputs):
        '''
        For batch_size=1, just return the last index.
        '''
        return np.ones(inputs.shape[0], dtype=np.int32) * (inputs.shape[1] - 1)

    def _encoder_args(self, entity_tokens):
        inputs = np.reshape(self.env.textint_map.text_to_int(entity_tokens, 'encoding'), [1, -1])
        encoder_args = {'inputs': inputs,
                'last_inds': self._get_last_inds(inputs),
                'init_state': self.encoder_state
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
            self.new_turn = False
        else:
            assert self.decoder_state is not None
            init_state = self.decoder_state
            start_symbol = markers.EOS

        inputs = np.reshape(self.env.textint_map.text_to_int([start_symbol], 'decoding'), [1, 1])

        decoder_args = self._decoder_args(init_state, inputs)
        decoder_output_dict = self.model.decoder.decode(sess, self.env.max_len, batch_size=1, stop_symbol=self.env.stop_symbol, **decoder_args)

        entity_tokens = self._pred_to_token(decoder_output_dict['preds'])
        self._update_states(sess, decoder_output_dict, entity_tokens)

        # Text message
        return [x if not is_entity(x) else x[0] for x in entity_tokens[0]]

    def encode(self, entity_tokens):
        encoder_args = self._encoder_args(entity_tokens)
        self.encoder_output_dict = self.model.encoder.encode(self.env.tf_session, **encoder_args)
        self.encoder_state = self.encoder_output_dict['final_state']
        self.new_turn = True

class GraphNeuralSession(RNNNeuralSession):
    def __init__(self, agent, kb, env):
        super(GraphNeuralSession, self).__init__(agent, kb, env)
        self.graph = GraphBatch([Graph(kb)])

        self.utterances = None
        self.context = None
        self.graph_data = None
        self.checklists = None

    def _match(self, item):
        for it in self.kb.items:
            if it == item:
                return it
        return None

    def _encoder_args(self, entity_tokens):
        encoder_args = super(GraphNeuralSession, self)._encoder_args(entity_tokens)
        graph_data = self.graph.get_batch_data([entity_tokens], None, self.utterances)
        encoder_args['entities'] = graph_data['encoder_entities']
        encoder_args['utterances'] = graph_data['utterances']
        encoder_args['graph_data'] = graph_data
        return encoder_args

    def _decoder_init_state(self, sess):
        self.checklists = self.graph.get_zero_checklists(1)
        init_state = self.model.decoder.compute_init_state(sess,
                self.encoder_state,
                self.encoder_output_dict['final_output'],
                self.encoder_output_dict['context'],
                self.checklists)
        return init_state

    def _decoder_args(self, init_state, inputs):
        decoder_args = super(GraphNeuralSession, self)._decoder_args(init_state, inputs)
        decoder_args['checklists'] = self.checklists
        decoder_args['graphs'] = self.graph
        decoder_args['vocab'] = self.env.vocab
        return decoder_args

    def _update_states(self, sess, decoder_output_dict, entity_tokens):
        # TODO: update context in decoder state; for now this is fine because it never "wait",
        # i.e. each turn has one utterance.
        self.decoder_state = decoder_output_dict['final_state']
        self.encoder_state = decoder_output_dict['final_state'][0]

        # Update graph and utterances
        graph_data = self.graph.get_batch_data(None, entity_tokens, self.utterances)
        self.utterances = self.model.decoder.update_utterances(sess, graph_data['decoder_entities'], decoder_output_dict['final_output'], graph_data['utterances'], graph_data)

    def _pred_to_token(self, preds):
        if self.env.copy:
            preds = self.graph.copy_preds(preds, self.env.vocab.size)
        entity_tokens = pred_to_token(preds, self.env.stop_symbol, self.env.remove_symbols, self.env.textint_map)
        # TODO: The output does not have surface form yet. Add the canonical form as surface for now.
        entity_tokens = [[(x[0], x) if is_entity(x) else x for x in toks] for toks in entity_tokens]
        return entity_tokens
