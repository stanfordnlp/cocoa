import random
import copy
import sys
import re
import operator
from collections import namedtuple, defaultdict

from cocoa.core.entity import is_entity

from session import Session
from core.tokenizer import tokenize

class RulebasedSession(object):
    @staticmethod
    def get_session(agent, kb, tracker, config=None):
        return BaseRulebasedSession(agent, kb, tracker, config)

Config = namedtuple('Config', ['target', 'bottomline', 'patience'])
default_config = Config(8, 5, 10)

class BaseRulebasedSession(Session):
    def __init__(self, agent, kb, lexicon, config):
        super(BaseRulebasedSession, self).__init__(agent)
        self.kb = kb

        self.lexicon = lexicon
        self.my_proposal = None
        self.their_proposal = None
        self.config = default_config if config is None else config

        self.state = {
                'selected': False,
                'my_action': None,
                'their_action': None,
                'num_utterance_sent': 0,
                'time': 0
                }

    def propose(self, proposal):
        self.state['proposed'] = True
        templates = [
                "i need {my_offer}, you can have the rest.",
                "how about i get {my_offer}?",
                "i would like {my_offer}.",
                "i'll take {my_offer}",
                "you get {their_offer}, i take the rest.",
                ]
        s = random.choice(templates).format(
                my_offer=self.offer_to_string(proposal[self.agent]),
                their_offer=self.offer_to_string(proposal[1-self.agent]))
        return self.message(s)

    def is_neg(self, tokens):
        for token in tokens:
            if token in ("nope", "not", "cannot", "n't", "sorry", "no"):
                return True
        return False

    def intro(self):
        self.state['my_action'] = 'intro'
        s = [  "What movies do you like to watch?",
                "What is your favorite action movie?",
                "Hi, what is your favorite movie?"
            ]
        return self.message(random.choice(s))

    def generic_response(self):
        self.state['my_action'] = 'counting'
        s = "this is my {} time talking".format(self.state['time']/2)
        return self.message(s)

    def receive(self, event):
        if event.action == 'done':
            self.state['num_utterance_sent'] = 0
            self.state['their_action'] = 'done'
        elif event.action == 'message':
            self.state['num_utterance_sent'] = 0
            self.state['time'] += 1

    def wait(self):
        return None

    def send(self):
        # Strict turn-taking
        if self.state['num_utterance_sent'] > 0:
            return self.wait()
        self.state['num_utterance_sent'] += 1

        if self.state['their_action'] == 'done':
            return self.done()

        self.state['time'] += 1
        # Actual dialog
        if self.state['time'] < 3:
            print("Tried to introduce")
            return self.intro()
        elif self.state['time'] >= 14:
            print("Tried to finish")
            return self.done()
        else:
            print("Tried to talk")
            return self.generic_response()

        raise Exception('Uncaught case')
