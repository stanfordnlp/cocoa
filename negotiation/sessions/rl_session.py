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
    def __init__(self, agent, kb, env, session):
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.model = env.model
        self.session = session  # likely PytorchSession, but could anything
        self.kb = kb
        self.rewards = []

        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        # self.batcher = self.env.dialogue_batcher
        # self.kb_context_batch = self.batcher.create_context_batch([self.dialogue], self.batcher.kb_pad)
        # self.max_len = 100

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
        tokens = self.generate()
        if tokens is None:
            return None
        self.dialogue.add_utterance(self.agent, list(tokens))
        tokens = self.map_prices(tokens)
        accepted = False

        if len(tokens) > 0:
            if tokens[0] == markers.OFFER:
                try:
                    return self.offer({'price': float(tokens[1])})
                except ValueError:
                    return None
            elif tokens[0] == markers.ACCEPT:
                self.update_reward(agree=True, self.offer)
                return self.accept()
            elif tokens[0] == markers.REJECT:
                self.update_reward(agree=False, self.offer)
                return self.reject()


        s = self.attach_punct(' '.join(tokens))
        return self.message(s)

    def update_reward(self, agree, new_reward):
        self.reward.append(new_reward)
        reward calculation is difference between normalized price offers
        if offers do not match, reward = 0
        if agent is buyer = any amount below 1.0 of value entity
        if agent is seller = any amount above 1.0 of value entity
        self.logprob = something

    @classmethod
    def get_gradient(cls, update(self, agree, reward):
        self.t += 1
        # reward = reward if agree else 0
        # self.all_rewards.append(reward)
        # standardize the reward
        # r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(self.reward))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r


        ''' ---- Move into reinforce.py controller ----
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.rl_clip)
        self.opt.step()
        '''





