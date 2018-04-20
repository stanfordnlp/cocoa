import random
import numpy as np
import pdb

import torch
from torch.autograd import Variable
import torch.nn as nn

from neural.preprocess import markers, Dialogue
from session import Session

class RLSession(Session):
    def __init__(self, agent, session, optim):
        super(RLSession, self).__init__(agent)
        assert hasattr(session, 'logprobs')
        self.session = session  # likely PytorchSession, but could anything
        self.optim = optim
        self.model = session.model
        self.kb = session.kb
        # TODO: read from cmd
        self.discount_factor = 0.9

    def receive(self, event):
        return self.session.receive(event)

    def send(self):
        return self.session.send()

    def update(self, reward):
        #self.t += 1
        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(reward))
        reward_collector = []
        for _ in self.session.logprobs:
            reward_collector.insert(0, g)
            g = g * self.discount_factor

        loss = 0
        for lp, rc in zip(self.session.logprobs, reward_collector):
            print type(lp), type(rc)
            loss -= lp * rc

        print 'reward:', reward
        print 'loss:', loss.data[0]
        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.model_args.rl_clip)
        self.optim.step()

