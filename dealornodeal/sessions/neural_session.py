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
from neural.symbols import markers

from fb_model import utils
from fb_model.agent import LstmRolloutAgent, LstmAgent

class FBNeuralSession(Session):
    """A wrapper for LstmRolloutAgent from Deal or No Deal
    Not used by for cocoa
    """
    def __init__(self, agent, kb, model, args):
        super(FBNeuralSession, self).__init__(agent)
        self.kb = kb
        self.model = LstmAgent(model, args)
        context = self.kb_to_context(self.kb)
        self.model.feed_context(context)
        self.state = {
                'selected': False,
                'quit': False,
                }

    def kb_to_context(self, kb):
        """Convert `kb` to context used by the LstmRolloutAgent.
        Returns: context (list[str]):
                [book_count, book_value, hat_count, hat_value, ball_count, ball_value]
        """
        context = []
        for item in ('book', 'hat', 'ball'):
            context.extend([str(kb.item_counts[item]), str(kb.item_values[item])])
        return context

    def parse_choice(self, choice):
        """Convert agent choice to dict.

        Args: choice (list[str]):
            e.g. ['item0=1', 'item1=2', 'item2=0', 'item0=0', 'item1=0', 'item2=2'],
            where item 0-2 are book, hat, ball.
            The agent's choice are the first 3 elements.

        Return: data (dict): {item: count}
        """
        try:
            return {item: int(choice[i].split('=')[1]) for i, item in enumerate(('book', 'hat', 'ball'))}
        except:
            print 'Cannot parse choice:', choice
            #return {item: 0 for i, item in enumerate(('book', 'hat', 'ball'))}
            return None

    def receive(self, event):
        if event.action == 'select':
            self.state['selected'] = True
        elif event.action == 'quit':
            self.state['quit'] = True
        elif event.action == 'message':
            tokens = event.data.lower().strip().split() + ['<eos>']
            self.model.read(tokens)

    def select(self):
        choice = self.model.choose()
        proposal = self.parse_choice(choice)
        if proposal is None:
            return self.quit()
        return super(FBNeuralSession, self).select(proposal)

    def _is_selection(self, out):
        return len(out) == 1 and out[0] == '<selection>'

    def send(self):
        if self.state['selected']:
            return self.select()

        if self.state['quit']:
            return self.quit()

        tokens = self.model.write()
        if self._is_selection(tokens):
            return self.select()
        # Omit the last <eos> symbol
        return self.message(' '.join(tokens[:-1]))

class NeuralSession(Session):
    def __init__(self, agent, kb, env):
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.kb = kb
        self.builder = env.utterance_builder
        self.generator = env.dialogue_generator
        self.cuda = env.cuda

        self.batcher = self.env.dialogue_batcher
        fake_outcome = {"item_split": [
                           {"book": 0, "hat": 0, "ball": 0},
                           {"book": 0, "hat": 0, "ball": 0}]
                       }
        self.dialogue = Dialogue(agent, kb, fake_outcome, None)
        self.max_len = 100

        self.dialogue.scenario_to_int()
        self.dialogue.selection_to_int()

        self.partner_quit = False

    # TODO: move this to preprocess?
    def convert_to_int(self):
        for i, turn in enumerate(self.dialogue.token_turns):
            for curr_turns, stage in izip(self.dialogue.turns, ('encoding', 'decoding', 'target')):
                if i >= len(curr_turns):
                    curr_turns.append(self.env.textint_map.text_to_int(turn, stage))
                else:
                    # Already converted
                    pass

    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        utterance = self.env.preprocessor.process_event(event, self.agent, None)
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
        return s

    def _get_proposal(self, tokens):
        try:
            proposal = {}
            for token in tokens[1:4]:
                ss = token.split('=')
                proposal[ss[0]] = int(ss[1])
        except ValueError:
            print tokens
            import sys; sys.exit()
        return proposal

    def send(self):
        if self.partner_quit:
            return self.quit()

        tokens = self.generate()
        if tokens is None:
            return None
        self.dialogue.add_utterance(self.agent, list(tokens))
        #tokens = self.builder.entity_to_str(tokens, self.kb)

        if len(tokens) > 0:
            if tokens[0] == markers.SELECT:
                proposal = self._get_proposal(tokens)
                #proposal = {item: int(count) for item, count in izip(self.items, tokens[1:4])}
                return self.select(proposal)
            elif tokens[0] == markers.QUIT:
                return self.quit()

        s = self.attach_punct(' '.join([str(x) for x in tokens]))
        #print 'send:', s
        return self.message(s)

class PytorchNeuralSession(NeuralSession):
    def __init__(self, agent, kb, env):
        super(PytorchNeuralSession, self).__init__(agent, kb, env)
        self.vocab = env.utterance_vocab
        self.kb_vocab = env.kb_vocab
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
                        'targets': np.copy(encoder_turns[0]),
                        'scenarios': np.array([self.dialogue.scenario]),
                        'selections': np.array([self.dialogue.selection]),
                    }

        context_data = {
                'agents': [self.agent],
                'kbs': [self.kb],
                }

        return Batch(encoder_args, decoder_args, context_data, self.vocab,
                sort_by_length=False, num_context=num_context, cuda=self.cuda)

    def _run_generator(self, batch, enc_state):
        output_data = self.generator.generate_batch(batch, gt_prefix=self.gt_prefix, enc_state=enc_state)
        return output_data

    def generate(self):
        if len(self.dialogue.agents) == 0:
            self.dialogue._add_utterance(1 - self.agent, [])
        batch = self._create_batch()
        enc_state = self.dec_state.hidden if self.dec_state is not None else None
        output_data = self._run_generator(batch, enc_state)

        if self.stateful:
            # TODO: only works for Sampler for now. cannot do beam search.
            self.dec_state = output_data['dec_states']
            self.dec_state = None
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
                          self.env.utterance_vocab,
                          num_context=Dialogue.num_context, cuda=self.env.cuda)
            yield batch

class PytorchLFNeuralSession(PytorchNeuralSession):
    def __init__(self, agent, kb, env):
        super(PytorchLFNeuralSession, self).__init__(agent, kb, env)
        self.items = env.preprocessor.lexicon.items
        self.item_counts = self.kb.item_counts
        self.my_proposal = None
        self.partner_proposal = None
        self.partner_select = False
        self.partner_quit = False

    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        if event.action == 'select':
            utterance = [markers.SELECT]
            self.partner_select = True
        elif event.action == 'quit':
            self.partner_quit = True
            utterance = None
        else:
            utterance = event.data.split()
            # Compute what they want for us
            if utterance[0] in ('propose', 'insist'):
                self.partner_proposal = self._get_proposal(utterance)
                utterance = ['propose'] + ['{item}={count}'.format(
                    item=item, count=(self.kb.item_counts[item] - self.partner_proposal[item]))
                    for item in self.items]
        # Empty message
        if utterance is None:
            return

        #print 'receive:', utterance
        self.dialogue.add_utterance(event.agent, utterance)

    def _run_generator(self, batch, enc_state):
        output_data = self.generator.generate_batch(batch,
                gt_prefix=self.gt_prefix, enc_state=enc_state, kb=self.kb, select=self.partner_select)
        return output_data

    # TODO: this should be consistent with preprocessor. better to be modular.
    def _proposal_to_tokens(self, proposal):
        tokens = ['{item}={count}'.format(item=item, count=proposal[item])
                for item in self.items]
        return tokens

    def generate(self):
        tokens = super(PytorchLFNeuralSession, self).generate()
        if tokens[0] in ('propose', 'insist'):
            self.my_proposal = self._get_proposal(tokens)
        elif tokens[0] == markers.SELECT:
            proposal = self._get_proposal(tokens)
            if self.partner_select and self.partner_proposal is not None:
                my_proposal = {item: self.item_counts[item] - count for
                        item, count in self.partner_proposal.iteritems()}
                tokens = [markers.SELECT] + self._proposal_to_tokens(my_proposal)
            elif not self.partner_select and self.my_proposal is not None:
                tokens = [markers.SELECT] + self._proposal_to_tokens(self.my_proposal)
        return tokens

