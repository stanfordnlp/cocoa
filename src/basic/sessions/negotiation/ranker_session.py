from src.basic.sessions.session import Session
from src.model.negotiation.preprocess import markers, Dialogue
from src.model.vocab import Vocabulary
from src.basic.entity import is_entity, Entity
from src.model.evaluate import pred_to_token
import numpy as np
import random
import re
from itertools import izip

class IRRankerSession(Session):
    """
    RankerSession represents a dialogue agent backed by a neural model. This class stores the knowledge graph,
    tensorflow session, and other model-specific information needed to update the model (when a message is received)
    or generate a response using the model (when send() is called).
    This class is closely related to but different from NeuralSystem. NeuralSystem is the class that actually loads the
    model from disk, while RankerSession represents a specific instantiation of that model, and contains information
    (like the agent index and the knowledge base) that are specific to an agent in the current dialogue.
    """
    def __init__(self, agent, kb, env):
        super(IRRankerSession, self).__init__(agent)
        self.env = env
        self.ranker = env.ranker
        self.retriever = env.retriever
        self.kb = kb
        self.role = self.kb.facts['personal']['Role']
        self.category = self.kb.facts['item']['Category']
        self.title = self.kb.facts['item']['Title']
        self.description = self.kb.facts['item']['Description']
        self.mentioned_entities = set()
        self.prev_turns = []
        self.cached_utterances = []
        self.replied = False
        self.offered = False
        #self.log = open('chat.debug.log', 'a')
        #self.log.write('-------------------------------------\n')

    def receive(self, event):
        if event.action == 'offer':
            self.offered = True
        # Parse utterance
        entity_tokens = self.env.preprocessor.process_event(event, self.agent, self.kb, mentioned_entities=self.mentioned_entities)
        # Empty message
        if entity_tokens is None:
            return

        for token in entity_tokens:
            if is_entity(token):
                self.mentioned_entities.add(token.canonical)

        self.replied = False

        self.prev_turns.append(entity_tokens)

        #entity_tokens += [markers.EOS]
        #self.encode(entity_tokens)

    def get_candidates(self):
        if len(self.prev_turns) == 0:
            self.prev_turns.append([markers.EOS])
        candidates = self.retriever.search(self.role, self.category, self.title, self.prev_turns)
        return candidates

    def select(self):
        candidates = self.get_candidates()
        batch = {'token_candidates': [candidates]}
        # batch_size = 1
        response = self.ranker.select(batch)[0]
        return response

    def split_sentences(self, tokens):
        '''
        Split sentences by EOS.
        '''
        utterances = []
        s = []
        for w in tokens:
            if w == markers.EOS:
                utterances.append(s)
                s = []
            else:
                s.append(w)
        return utterances

    def send(self):
        # Don't send consecutive utterances
        if self.replied:
            return None

        if self.offered:
            return self.accept()

        #if len(self.cached_utterances) > 0:
        #    tokens = self.cached_utterances.pop(0)
        #else:
        #    tokens = self.select()
        #    self.prev_turns.append(tokens)
        #    utterances = self.split_sentences(tokens)
        #    tokens = utterances[0]
        #    self.cached_utterances.extend(utterances[1:])

        tokens = self.select()
        self.prev_turns.append(tokens)

        for token in tokens:
            if is_entity(token):
                self.mentioned_entities.add(token.canonical)

        # TODO: handle multiple sentences properly
        for i, token in enumerate(tokens):
            if token == markers.OFFER:
                try:
                    return self.offer({'price': float(tokens[i+1].canonical.value)})
                except ValueError:
                    pass
        if len(tokens) > 1:
            if tokens[0] == markers.QUIT:
                return self.quit()
            elif tokens[0] == markers.ACCEPT:
                return self.accept()
            elif tokens[0] == markers.REJECT:
                return self.reject()
        # TODO: handle price properly
        s = ' '.join(['_price_' if is_entity(x) else x for x in tokens if x != markers.EOS])
        self.replied = True
        return self.message(s)

class EncDecRankerSession(IRRankerSession):
    def __init__(self, agent, kb, env):
        super(EncDecRankerSession, self).__init__(agent, kb, env)
        self.encoder_state = None
        self.context_batch = self._get_int_context()
        self.GO = env.mappings['vocab'].to_ind(markers.GO_S if self.role == 'seller' else markers.GO_B)
        self.PAD = env.mappings['vocab'].to_ind(markers.PAD)

    def _get_int_context(self):
        category = self.env.mappings['cat_vocab'].to_ind(self.category)
        title = map(self.env.mappings['kb_vocab'].to_ind, self.title)
        description = map(self.env.mappings['kb_vocab'].to_ind, self.description)
        context = {
                'category': np.array(category).reshape([1, -1]),
                'title': np.array(title).reshape([1, -1]),
                'description': np.array(description).reshape([1, -1]),
                }
        return context

    def _get_int_candidates(self, token_candidates):
        candidates = [self.env.textint_map.text_to_int(c['response'], 'decoding') for c in token_candidates if 'response' in c]
        num_candidates = len(candidates)
        max_len = max([len(c) for c in candidates])
        T = np.full([num_candidates, max_len+1], self.PAD, dtype=np.int32)
        T[:, 0] = self.GO
        for i, c in enumerate(candidates):
            T[i, 1:len(c)+1] = c
        T = T.reshape(1, num_candidates, -1)  # batch_size = 1
        return T

    def _process_entity_tokens(self, entity_tokens):
        int_inputs = np.reshape(self.env.textint_map.text_to_int(entity_tokens), [1, -1])
        return int_inputs

    def _ranker_args(self, candidates):
        decoder_args = {
                'inputs': None,
                'targets': None,
                'context': self.context_batch,
                'init_state': self.encoder_state,
                }
        kwargs = {
                'decoder': decoder_args,
                }
        return kwargs

    def _encoder_args(self, entity_tokens):
        inputs = self._process_entity_tokens(entity_tokens)
        #self.log.write('encoder entities:%s\n' % str(entities))
        encoder_args = {'inputs': inputs,
                'init_state': self.encoder_state,
                }
        return encoder_args

    def encode(self, entity_tokens):
        encoder_args = self._encoder_args(entity_tokens)
        #self.log.write('encode:%s\n' % str(entity_tokens))
        encoder_output_dict = self.ranker.model.encoder.run_encode(self.env.tf_session, **encoder_args)
        self.encoder_state = encoder_output_dict['final_state']
        #self.new_turn = True

    def receive(self, event):
        if event.action == 'offer':
            self.offered = True
        # Parse utterance
        entity_tokens = self.env.preprocessor.process_event(event, self.agent, self.kb, mentioned_entities=self.mentioned_entities)
        # Empty message
        if entity_tokens is None:
            return

        for token in entity_tokens:
            if is_entity(token):
                self.mentioned_entities.add(token.canonical)

        self.replied = False

        self.prev_turns.append(entity_tokens)

        entity_tokens += [markers.EOS]
        self.encode(entity_tokens)

    def select(self):
        if self.encoder_state is None:
            self.encode([markers.EOS])
        token_candidates = self.get_candidates()
        candidates = self._get_int_candidates(token_candidates)
        kwargs = self._ranker_args(candidates)
        candidates_loss, final_states = self.ranker.score(candidates, kwargs=kwargs, states=True)
        best_candidate = self.ranker.sample_candidates(candidates_loss)[0]
        self.encoder_state = final_states[best_candidate]
        return token_candidates[best_candidate]['response']

