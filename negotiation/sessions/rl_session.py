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
        super(RLSession, self).__init__(agent, kb, env)
        self.session = session  # likely PytorchSession, but could anything
        self.logprobs = []
        # self.trainable = trainable

    def generate(self):
        batch = self.session._create_batch()
        encoder_init_state = None

        output_data = self.generator.generate_batch(batch, gt_prefix=self.gt_prefix)
        entity_tokens = self.output_to_tokens(output_data)
        log_probability = self.calculate_logprob(output_data)
        self.logprobs.append(log_probability)

        if not self._is_valid(entity_tokens):
            return None
        return entity_tokens

    def calculate_logprob(self, data):
        # generator._from_beam() method already calculated scores
        prob = F.softmax(data['scores'][0])
        logprob = F.log_softmax(data['scores'][0])

        word = prob.multinomial().detach()
        logprob = logprob.gather(0, word)

        return logprob

    @classmethod
    def apply_update(cls, reward):
        self.t += 1
        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(self.reward))
        reward_collector = []
        for _ in self.logprobs:
            reward_collector.insert(0, g)
            g = g * self.model_args.discount_factor

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, rc in zip(self.logprobs, reward_collector):
            loss -= lp * rc

        # if self.trainable == False:
        #     param.requires_grad = False
        self.model.optim.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.model.parameters(), self.model_args.rl_clip)
        self.model.optim.step()

# Pure Wrapper Version
class RLSession(Session):
    def __init__(self, agent, kb, env, session, trainable=True):
        super(RLSession, self).__init__(agent)
        self.env = env
        self.model = env.model
        self.kb = kb

        self.batcher = self.env.dialogue_batcher
        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        self.kb_context_batch = self.batcher.create_context_batch([self.dialogue], self.batcher.kb_pad)
        self.max_len = 100

    def generate(self):
        batch = self.session._create_batch()
        encoder_init_state = None

        output_data = self.generator.generate_batch(batch, gt_prefix=self.gt_prefix)
        entity_tokens = self.output_to_tokens(output_data)
        log_probability = self.calculate_logprob(output_data)
        self.logprobs.append(log_probability)

        if not self._is_valid(entity_tokens):
            return None
        return entity_tokens

    def calculate_logprob(self, data):
        # generator._from_beam() method already calculated scores
        prob = F.softmax(data['scores'][0])
        logprob = F.log_softmax(data['scores'][0])

        word = prob.multinomial().detach()
        logprob = logprob.gather(0, word)

        return logprob

    @classmethod
    def apply_update(cls, reward):
        self.t += 1
        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(self.reward))
        reward_collector = []
        for _ in self.logprobs:
            reward_collector.insert(0, g)
            g = g * self.model_args.discount_factor

        loss = 0
        for lp, rc in zip(self.logprobs, reward_collector):
            loss -= lp * rc

        self.model.optim.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.model.parameters(), self.model_args.rl_clip)
        self.model.optim.step()

    def convert_to_int(self):
        self.session.convert_to_int()

    def receive(self, event):
        self.session.receive(event)

    def _has_entity(self, tokens):
        return self._has_entity(tokens)

    def attach_punct(self, s):
        return self.session.attach_punct(s)

    def map_prices(self, entity_tokens):
        return self.session(entity_tokens)

    def send(self):
        return self.session.send()

    def get_decoder_inputs(self):
        return self.session.get_decoder_inputs()

    def _create_batch(self):
        return self.session._create_batch()

    def generate(self):
        return self.session.generate()

    def _is_valid(self, tokens):
        return self.session._is_valid(tokens)

    def output_to_tokens(self, data):
        return self.session.output_to_tokens(data)
