from src.basic.sessions.session import Session
from src.model.negotiation.preprocess import markers, Dialogue
from src.model.vocab import Vocabulary
from src.basic.entity import is_entity, Entity
from src.model.evaluate import pred_to_token
import numpy as np
import random
import re
from itertools import izip

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
        self.role = self.kb.facts['personal']['Role']
        self.mentioned_entities = set()
        #self.log = open('chat.debug.log', 'a')
        #self.log.write('-------------------------------------\n')

    def encode(self, entity_tokens):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    def receive(self, event):
        # Parse utterance
        entity_tokens = self.env.preprocessor.process_event(event, self.agent, self.kb, mentioned_entities=self.mentioned_entities)
        print entity_tokens
        # Empty message
        if entity_tokens is None:
            return

        for token in entity_tokens:
            if is_entity(token):
                self.mentioned_entities.add(token.canonical)
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
        return tokens

    def attach_punct(self, s):
        s = re.sub(r' ([.,!?;])', r'\1', s)
        s = re.sub(r'\.{3,}', r'...', s)
        return s

    def send(self):
        # Don't send consecutive utterances with entities
        if self.sent_entity and not self.env.consecutive_entity:
            return None
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
                self.mentioned_entities.add(token.canonical)
        if self.env.realizer is None:
            tokens = [x if not is_entity(x) else x.surface for x in tokens]
        else:
            tokens = self.env.realizer.realize_entity(tokens)
        if len(tokens) > 1 and tokens[0] == markers.OFFER:
            try:
                self.offer(float(tokens[1]))
            except ValueError:
                pass
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

    @classmethod
    def _get_last_inds(cls, inputs):
        '''
        For batch_size=1, just return the last index.
        '''
        return np.ones(inputs.shape[0], dtype=np.int32) * (inputs.shape[1] - 1)

    def _process_entity_tokens(self, entity_tokens, stage):
        int_inputs = np.reshape(self.env.textint_map.text_to_int(entity_tokens), [1, -1])
        return int_inputs
        #inputs, entities = self.env.textint_map.process_entity(int_inputs, stage)
        #return inputs, entities

    def _encoder_args(self, entity_tokens):
        inputs = self._process_entity_tokens(entity_tokens, 'encoding')
        #self.log.write('encoder entities:%s\n' % str(entities))
        encoder_args = {'inputs': inputs,
                'last_inds': self._get_last_inds(inputs),
                'init_state': self.encoder_state,
                }
        return encoder_args

    def _decoder_init_state(self, sess):
        return self.encoder_state

    def _decoder_args(self, entity_tokens):
        inputs = self._process_entity_tokens(entity_tokens, 'decoding')
        decoder_args = {'inputs': inputs,
                'last_inds': self._get_last_inds(inputs),
                'init_state': self.decoder_state,
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
            start_symbol = markers.GO_S if self.role == 'seller' else markers.GO_B
            self.decoder_state = self._decoder_init_state(sess)
        else:
            assert self.decoder_state is not None
            start_symbol = markers.EOS

        entity_tokens = [start_symbol]
        decoder_args = self._decoder_args(entity_tokens)
        decoder_output_dict = self.model.decoder.run_decode(sess, self.env.max_len, batch_size=1, stop_symbol=self.env.stop_symbol, **decoder_args)

        # TODO: why [0]
        entity_tokens = self._pred_to_token(decoder_output_dict['preds'])[0]
        if not self._is_valid(entity_tokens):
            return None
        #self.log.write('decode:%s\n' % str(entity_tokens))
        self._update_states(sess, decoder_output_dict, entity_tokens)

        # Text message
        if self.new_turn:
            self.new_turn = False
        return entity_tokens

    def _is_valid(self, tokens):
        if not tokens:
            return False
        if Vocabulary.UNK in tokens:
            return False
        return True

    def encode(self, entity_tokens):
        encoder_args = self._encoder_args(entity_tokens)
        #self.log.write('encode:%s\n' % str(entity_tokens))
        self.encoder_output_dict = self.model.encoder.run_encode(self.env.tf_session, **encoder_args)
        self.encoder_state = self.encoder_output_dict['final_state']
        self.new_turn = True

    def _pred_to_token(self, preds):
        entity_tokens, _ = pred_to_token(preds, self.env.stop_symbol, self.env.remove_symbols, self.env.textint_map)
        # NOTE: entities are CanonicalEntities, change to Entity
        entity_tokens = [[Entity(str(x.value), x) if is_entity(x) else x for x in toks] for toks in entity_tokens]
        entity_tokens = [Dialogue.original_price(self.kb, toks) for toks in entity_tokens]
        return entity_tokens
