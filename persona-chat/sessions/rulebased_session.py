import random
import copy
import sys
import re
import operator
from collections import namedtuple, defaultdict

from cocoa.core.entity import is_entity
from cocoa.sessions.rulebased_session import RulebasedSession as BaseRulebasedSession
from cocoa.model.parser import LogicalForm as LF

from model.parser import Parser, Utterance
from model.dialogue_state import DialogueState
from core.tokenizer import tokenize

class RulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        parser = Parser(agent, kb, lexicon)
        state = DialogueState(agent, kb)
        super(RulebasedSession, self).__init__(agent, kb, parser, generator, manager, state, sample_temperature=5.)

        self.kb = kb
        self.personas = kb.personas

    def receive(self, event):
        super(RulebasedSession, self).receive(event)

    def select(self, split):
        print "came to rulebased select, should not happen"
        utterance = Utterance(logical_form=LF('select', split=split))
        self.state.update(self.agent, utterance)
        metadata = self.metadata(utterance)
        return super(RulebasedSession, self).select(split, metadata=metadata)

    def send(self):
        action = self.manager.choose_action(state=self.state)
        print("action: {}".format(action) )
        if action == 'done':
            return self.done()
        else:
            lf = LF("placeholder_intent")
            text = random.choice(self.personas)
            utterance = Utterance(raw_text=text, logical_form=lf)
            return self.message(utterance)

        raise Exception('Uncaught case')