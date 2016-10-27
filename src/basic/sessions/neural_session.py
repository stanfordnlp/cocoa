__author__ = 'anushabala'
from session import Session
from model.graph import Graph, GraphBatch
from model.preprocess import EOT, EOS, GO, SELECT, tokenize
from model.vocab import is_entity
from model.evaluate import pred_to_token
import numpy as np

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
        self.graph = GraphBatch([Graph(kb)])

        self.matched_item = None
        self.encoder_state = None  # Starting state for generation/encoding
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
        return np.ones_like(inputs, dtype=np.int32) * (inputs.shape[1] - 1)

    def encode(self, entity_tokens):
        graph_data = self.graph.get_batch_data(entity_tokens, None, self.utterances)
        # Convert inputs to integers
        inputs = np.reshape(self.env.textint_map.text_to_int(entity_tokens), [1, -1], dtype=np.int32)
        if self.env.copy:
            inputs = self.graphs.entity_to_vocab(inputs, self.env.vocab)
        last_inds = self._get_last_inds(inputs)
        feed_dict = self.model.update_feed_dict(encoder_inputs=inputs,
                encoder_inputs_last_inds=last_inds,
                encoder_init_state=self.encoder_state,
                encoder_entities=graph_data['encoder_entities'],
                encoder_input_utterances=graph_data['utterances'])
        self.model.add_graph_data(feed_dict, graph_data)
        self.graph_data = graph_data
        [self.encoder_state, self.utterances, self.context] = self.env.tf_session.run([self.model.encoder_final_state, self.model.encoder_output_utterances, self.model.encoder_output_context], feed_dict=feed_dict)

    def decode(self):
        if self.encoder_state is None:
            self.encode([[EOS]])
        inputs = np.reshape(self.textint_map.text_to_int([GO]), [1, 1], dtype=np.int32)
        last_inds = self._get_last_inds(inputs)
        # Continue from the encoder state and context (graph embedding)
        feed_dict = self.model.update_feed_dict(decoder_inputs=inputs,
                decoder_inputs_last_inds=last_inds,
                encoder_final_state=self.encoder_state,
                encoder_output_context=self.context)

        # Generate max_len steps
        preds = np.zeros([1, self.max_len], dtype=np.int32)
        final_state = None
        outputs = None
        for i in xrange(self.max_len):
            logits, final_state, outputs = sess.run([self.model.logits, self.model.decoder_final_state, self.model.decoder_outputs], feed_dict=feed_dict)
            step_decoder_inputs = self.model.get_prediction(logits)
            preds[:, [i]] = step_decoder_inputs
            if self.env.copy:
                step_decoder_inputs = self.graph.copy_preds(step_decoder_inputs, self.env.vocab.size)
                step_decoder_inputs = self.graph.entity_to_vocab(step_decoder_inputs, self.env.vocab)
            feed_dict = self.update_feed_dict(decoder_inputs=step_decoder_inputs,
                    decoder_inputs_last_inds=last_inds,
                    decoder_init_state=final_state)
        self.encoder_state = final_state[0]

        # Convert integers to tokens
        if self.env.copy:
            preds = graphs.copy_preds(preds, self.env.vocab.size)
        entity_tokens = pred_to_token(preds, self.env.stop_symbol, self.env.remove_symbols, self.env.textint_map)

        # Update graph and utterances
        graph_data = self.graph.get_batch_data(None, entity_tokens, self.utterances)
        feed_dict = self.model.get_feed_dict(decoder_entities=graph_data['decoder_entities'],
                decoder_input_utterances=graph_data['utterances'],
                decoder_outputs=outputs)
        self.model.add_graph_data(feed_dict, graph_data)
        [self.utterances] = self.env.tf_session.run([self.model.decoder_output_utterances], feed_dict=feed_dict)

        # Text message
        message = ' '.join(entity_tokens[0])
        return message

    def receive(self, event):
        # Parse utterance
        if event.action == 'select':
            self.matched_item = self._match(event.data)
            if self.matched_item is None:
                entity_tokens = [SELECT]
            else:
                # Got a match; we're done.
                return
        elif event.action == 'message':
            entity_tokens = self.lexicon.entitylink(tokenize(event.data))
        else:
            raise ValueError('Unknown event action.')
        entity_tokens += [EOS]

        self.encode([entity_tokens])

    def send(self):
        if self.matched_item is not None:
            return self.select(self.matched_item)
        return self.message(self.decode())
