from session import Session

from fb_model import utils
from fb_model.agent import LstmRolloutAgent

class FBNeuralSession(Session):
    """A wrapper for LstmRolloutAgent.
    """
    def __init__(self, agent, kb, model, args):
        super(NeuralSession, self).__init__(agent)
        self.kb = kb
        self.model = LstmRolloutAgent(model, args)
        context = self.kb_to_context(self.kb)
        self.model.feed_context(context)
        self.state = {
                'selected': False,
                'rejected': False,
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
            return {item: 0 for i, item in enumerate(('book', 'hat', 'ball'))}

    def receive(self, event):
        if event.action == 'select':
            self.state['selected'] = True
        elif event.action == 'reject':
            self.state['rejected'] = True
        elif event.action == 'message':
            tokens = event.data.lower().strip().split() + ['<eos>']
            self.model.read(tokens)

    def select(self):
        choice = self.model.choose()
        return super(NeuralSession, self).select(self.parse_choice(choice))

    def _is_selection(self, out):
        return len(out) == 1 and out[0] == '<selection>'

    def send(self):
        if self.state['selected']:
            return self.select()

        if self.state['rejected']:
            return self.reject()

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
        self.dialogue = Dialogue(agent, kb, None)
        self.max_len = 100

    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        # Parse utterance
        utterance = self.env.preprocessor.process_event(event, self.kb)
        # Empty message
        if utterance is None:
            return
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

    def send(self):
        tokens = self.generate()
        if tokens is None:
            return None
        self.dialogue.add_utterance(self.agent, list(tokens))
        tokens = self.builder.entity_to_str(tokens, self.kb)

        if len(tokens) > 0:
            if tokens[0] == markers.SELECT:
                return self.select()
            elif tokens[0] == markers.QUIT:
                return self.quit()

        s = self.attach_punct(' '.join(tokens))
        #print 'send:', s
        return self.message(s)

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
        self.dialogue.convert_to_int()
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
                    }

        context_data = {
                'agents': [self.agent],
                'kbs': [self.kb],
                }

        return Batch(encoder_args, decoder_args, context_data, self.vocab,
                sort_by_length=False, num_context=num_context, cuda=self.cuda)

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

        #if not self._is_valid(entity_tokens):
        #    return None
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

    # To support REINFORCE
    # TODO: this should be in NeuralSession
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

