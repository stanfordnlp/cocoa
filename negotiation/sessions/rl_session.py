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
from session import Session
from torch.autograd import Variable

class RLSession(Session):
    def __init__(self, agent, kb, env, session):
        super(RLSession, self).__init__(agent)
        assert hasattr(session, 'logprobs')
        self.session = session  # likely PytorchSession, but could anything

    def receive(self, event):
        return self.session.receive(event)

    def send(self):
        return self.session.send()

    @classmethod
    def update(cls, reward):
        self.t += 1
        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(self.reward))
        reward_collector = []
        for _ in self.session.logprobs:
            reward_collector.insert(0, g)
            g = g * self.model_args.discount_factor

        loss = 0
        for lp, rc in zip(self.logprobs, reward_collector):
            loss -= lp * rc

        self.model.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.model.parameters(), self.model_args.rl_clip)
        self.optim.step()

