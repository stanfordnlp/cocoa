__author__ = 'anushabala'
from session import Session
from src.model.graph import Graph, GraphBatch
from src.model.encdec import GraphEncoderDecoder
from src.model.preprocess import markers, entity_to_vocab
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
        # TODO: separate classes
        if isinstance(self.model, GraphEncoderDecoder):
            self.graph = GraphBatch([Graph(kb)])
        else:
            self.graph = None

        self.matched_item = None
        # Current encoder state, starting state for encoder
        self.encoder_state = None
        self.encoder_output_dict = None
        # Current encoder output, initial output for encoder (only used in attention models)
        self.encoder_output = None
        self.utterances = None
        self.context = None
        self.graph_data = None

    def _match(self, item):
        for it in self.kb.items:
            if it == item:
                return it
        return None

    @classmethod
    def _get_last_inds(cls, inputs):
        '''
        For batch_size=1, just return the last index.
        '''
        return np.ones(inputs.shape[0], dtype=np.int32) * (inputs.shape[1] - 1)

    def encode(self, entity_tokens):
        # Convert inputs to integers
        inputs = np.reshape(self.env.textint_map.text_to_int(entity_tokens), [1, -1])
        if self.env.copy:
            inputs = entity_to_vocab(inputs, self.env.vocab)

        encoder_args = {'inputs': inputs,
                'last_inds': self._get_last_inds(inputs),
                'init_state': self.encoder_state
                }
        if self.graph is not None:
            graph_data = self.graph.get_batch_data([entity_tokens], None, self.utterances)
            encoder_args['entities'] = graph_data['encoder_entities']
            encoder_args['utterances'] = graph_data['utterances']
            encoder_args['graph_data'] = graph_data
        self.encoder_output_dict = self.model.encoder.encode(self.env.tf_session, **encoder_args)

    def decode(self):
        sess = self.env.tf_session

        if self.encoder_output_dict is None:
            self.encode([markers.EOS])
        inputs = np.reshape(self.env.textint_map.text_to_int([markers.GO]), [1, 1])

        decoder_args = {'inputs': inputs,
                'last_inds': np.zeros([1], dtype=np.int32),
                'init_state': self.encoder_output_dict['final_state']
                }
        if self.graph is not None:
            decoder_args['init_state'] = self.model.decoder.compute_init_state(sess,
                    self.encoder_output_dict['final_state'],
                    self.encoder_output_dict['final_output'],
                    self.encoder_output_dict['context'])
            if self.env.copy:
                decoder_args['graphs'] = self.graph
                decoder_args['vocab'] = self.env.vocab
        decoder_output_dict = self.model.decoder.decode(sess, self.env.max_len, 1, **decoder_args)

        # TODO: separate!
        if self.graph is not None:
            self.encoder_state = decoder_output_dict['final_state'][0]
        else:
            self.encoder_state = decoder_output_dict['final_state']

        # Convert integers to tokens
        preds = decoder_output_dict['preds']
        if self.env.copy:
            preds = self.graph.copy_preds(preds, self.env.vocab.size)
        entity_tokens = pred_to_token(preds, self.env.stop_symbol, self.env.remove_symbols, self.env.textint_map)

        # Update graph and utterances
        if self.graph is not None:
            graph_data = self.graph.get_batch_data(None, entity_tokens, self.utterances)
            self.utterances = self.model.decoder.update_utterances(sess, graph_data['decoder_entities'], decoder_output_dict['final_output'], graph_data['utterances'], graph_data)

        # Text message
        message = ' '.join([x if not is_entity(x) else x[0] for x in entity_tokens[0]])
        return message

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
        else:
            raise ValueError('Unknown event action.')
        entity_tokens += [markers.EOS]

        print 'INPUT:', entity_tokens
        self.encode(entity_tokens)

    def send(self):
        if self.matched_item is not None:
            return self.select(self.matched_item)
        if random.random() < 0.5:  # Wait randomly
            return None
        return self.message(self.decode())
