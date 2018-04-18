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





