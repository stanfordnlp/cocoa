import random
import re
from itertools import izip
import numpy as np
import pdb

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity
from cocoa.pt_model.util import use_gpu

from core.event import Event
from neural.preprocess import markers, Dialogue
from neural.evaluator import Evaluator, add_evaluator_arguments
from sessions.neural_session import NeuralSession

class RLSession(NeuralSession):
    def __init__(self, agent, kb, env, session, trainable=True):
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.model = env.model
        self.session = session  # likely PytorchSession, but could anything
        self.trainable = trainable
        self.kb = kb
        self.logprobs = []

        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        # self.batcher = self.env.dialogue_batcher
        # self.kb_context_batch = self.batcher.create_context_batch([self.dialogue], self.batcher.kb_pad)
        # self.max_len = 100

    def write(self):
        # TODO: make new batch creation method that plays nicely with RL
        batch = self._create_batch()
        encoder_init_state = None

        output_data = self.generator.generate_batch(batch, gt_prefix=self.gt_prefix)
        entity_tokens = self.output_to_tokens(output_data)
        log_probability = self.calculate_logprob(output_data)

        if not self._is_valid(entity_tokens):
            return None
        return entity_tokens, log_probability

    def calculate_logprob(self, data):
        # generator._from_beam() method already calculated scores
        prob = F.softmax(data['scores'][0])
        logprob = F.log_softmax(data['scores'][0])

        word = prob.multinomial().detach()
        logprob = logprob.gather(0, word)

        return logprob


    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        # Parse utterance
        utterance = self.env.preprocessor.process_event(event, self.kb)
        # Empty message
        if utterance is None:
            return

        print 'receive:', utterance
        self.dialogue.add_utterance(event.agent, utterance)

    def send(self):
        # created new session method called write(), which behaves similar
        # to generate() except it also calculates logprob
        tokens, logprob = self.write()
        self.logprobs.append(logprob)

        if tokens is None:
            return None
        self.dialogue.add_utterance(self.agent, list(tokens))
        tokens = self.map_prices(tokens)

        if len(tokens) > 0:
            if tokens[0] == markers.OFFER:
                try:
                    return self.offer({'price': float(tokens[1])})
                except ValueError:
                    return None
            elif tokens[0] == markers.ACCEPT:
                return self.accept()
            elif tokens[0] == markers.REJECT:
                return self.reject()


        s = self.attach_punct(' '.join(tokens))
        return self.message(s)

    @classmethod
    def apply_update(cls, reward):
        self.t += 1
        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(self.reward))
        reward_collector = []
        for _ in self.logprobs:
            reward_collector.insert(0, g)
            g = g * self.model_args.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, rc in zip(self.logprobs, reward_collector):
            loss -= lp * rc

        if self.trainable == False:
            param.requires_grad = False

        self.model.optim.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.model.parameters(), self.model_args.rl_clip)
        self.model.optim.step()





