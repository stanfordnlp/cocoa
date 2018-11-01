import random
import re
from itertools import izip
import numpy as np
import torch
from onmt.Utils import use_gpu

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity

from core.event import Event
from session import Session
from neural.preprocess import markers, Dialogue
from neural.batcher import Batch

class NeuralSession(Session):
    def __init__(self, agent, kb, env):
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.kb = kb
        self.builder = env.utterance_builder
        self.generator = env.dialogue_generator
        self.cuda = env.cuda

        self.batcher = self.env.dialogue_batcher
        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        self.kb_context_batch = self.batcher.create_context_batch([self.dialogue], self.batcher.kb_pad)
        self.max_len = 100

    # TODO: move this to preprocess?
    def convert_to_int(self):
        for i, turn in enumerate(self.dialogue.token_turns):
            for curr_turns, stage in izip(self.dialogue.turns, ('encoding', 'decoding', 'target')):
                if i >= len(curr_turns):
                    curr_turns.append(self.env.textint_map.text_to_int(turn, stage))

    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        # Parse utterance
        utterance = self.env.preprocessor.process_event(event, self.kb)
        # Empty message
        if utterance is None:
            return

        #print 'receive:', utterance
        self.dialogue.add_utterance(event.agent, utterance)

    def _has_entity(self, tokens):
        for token in tokens:
            if is_entity(token):
                return True
        return False

    def attach_punct(self, s):
        s = re.sub(r' ([.,!?;])', r'\1', s)
        s = re.sub(r'\.{3,}', r'...', s)
        s = re.sub(r" 's ", r"'s ", s)
        s = re.sub(r" n't ", r"n't ", s)
        return s

    def send(self):
        tokens = self.generate()
        if tokens is None:
            return None
        self.dialogue.add_utterance(self.agent, list(tokens))

        if len(tokens) > 1 and tokens[0] == markers.OFFER and is_entity(tokens[1]):
            try:
                price = self.builder.get_price_number(tokens[1], self.kb)
                return self.offer({'price': price})
            except ValueError:
                #return None
                pass
        tokens = self.builder.entity_to_str(tokens, self.kb)

        if len(tokens) > 0:
            if tokens[0] == markers.ACCEPT:
                return self.accept()
            elif tokens[0] == markers.REJECT:
                return self.reject()
            elif tokens[0] == markers.QUIT:
                return self.quit()

        s = self.attach_punct(' '.join(tokens))
        #print 'send:', s
        return self.message(s)

    def iter_batches(self):
        """Compute the logprob of each generated utterance.
        """
        self.convert_to_int()
        batches = self.batcher.create_batch([self.dialogue])
        yield len(batches)
        for batch in batches:
            # TODO: this should be in batcher
            batch = Batch(batch['encoder_args'],
                          batch['decoder_args'],
                          batch['context_data'],
                          self.env.vocab,
                          num_context=Dialogue.num_context, cuda=self.env.cuda)
            yield batch


class PytorchNeuralSession(NeuralSession):
    def __init__(self, agent, kb, env):
        super(PytorchNeuralSession, self).__init__(agent, kb, env)
        self.vocab = env.vocab
        self.gt_prefix = env.gt_prefix

        self.dec_state = None
        self.stateful = self.env.model.stateful

        self.new_turn = False
        self.end_turn = False

    def get_decoder_inputs(self):
        # Don't include EOS
        utterance = self.dialogue._insert_markers(self.agent, [], True)[:-1]
        inputs = self.env.textint_map.text_to_int(utterance, 'decoding')
        inputs = np.array(inputs, dtype=np.int32).reshape([1, -1])
        return inputs

    def _create_batch(self):
        num_context = Dialogue.num_context

        # All turns up to now
        self.convert_to_int()
        encoder_turns = self.batcher._get_turn_batch_at([self.dialogue], Dialogue.ENC, None)

        encoder_inputs = self.batcher.get_encoder_inputs(encoder_turns)
        encoder_context = self.batcher.get_encoder_context(encoder_turns, num_context)
        encoder_args = {
                        'inputs': encoder_inputs,
                        'context': encoder_context
                    }
        decoder_args = {
                        'inputs': self.get_decoder_inputs(),
                        'context': self.kb_context_batch,
                        'targets': np.copy(encoder_turns[0]),
                    }

        context_data = {
                'agents': [self.agent],
                'kbs': [self.kb],
                }

        return Batch(encoder_args, decoder_args, context_data,
                self.vocab, sort_by_length=False, num_context=num_context, cuda=self.cuda)

    def generate(self):
        if len(self.dialogue.agents) == 0:
            self.dialogue._add_utterance(1 - self.agent, [])
        batch = self._create_batch()

        enc_state = self.dec_state.hidden if self.dec_state is not None else None
        output_data = self.generator.generate_batch(batch, gt_prefix=self.gt_prefix, enc_state=enc_state)

        if self.stateful:
            # TODO: only works for Sampler for now. cannot do beam search.
            self.dec_state = output_data['dec_states']
        else:
            self.dec_state = None

        entity_tokens = self._output_to_tokens(output_data)

        return entity_tokens

    def _is_valid(self, tokens):
        if not tokens:
            return False
        if Vocabulary.UNK in tokens:
            return False
        return True

    def _output_to_tokens(self, data):
        predictions = data["predictions"][0][0]
        tokens = self.builder.build_target_tokens(predictions, self.kb)
        return tokens

