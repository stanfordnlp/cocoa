import random
import copy
import sys
import re
import operator
from collections import namedtuple, defaultdict
import numpy as np

from cocoa.core.entity import is_entity, Entity, CanonicalEntity
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
        self.title_scores = self.score_titles()
        for k, v in self.title_scores.iteritems():
            print k, v

    def score_titles(self):
        titles = set([x for x in self.generator.iter_titles()])
        context_title_counts = defaultdict(int)
        for title in self.generator.iter_context_titles():
            context_title_counts[title] += 1
        #print titles
        #print context_title_counts
        title_scores = {t: c for t, c in context_title_counts.iteritems() if t in titles}
        return title_scores

    #def is_plot(self, tokens):
    #    s = ' '.join(tokens)
    #    if re.search(r'(movie|show|story|film|drama) about', s):
    #        return True
    #    if re.search(r'about ?', s):
    #        return True
    #    return False

    #def is_opinion(self, tokens):
    #    s = ' '.join(tokens)
    #    if re.search(r'(do|did) you (like|think|enjoy|feel)', s):
    #        return True
    #    return False

    def choose_title(self):
        titles = [(t, s) for t, s in self.title_scores.iteritems() if not t in self.state.mentioned_titles]
        if not titles:
            titles = self.title_scores.keys()
            title = random.choice(titles)
        else:
            title = random.choice(titles)[0]
        return title

    def inform_title(self):
        intent = 'inform-new-title'
        title = self.choose_title()
        print 'chosen title:', title
        template = self.retrieve_response_template(intent, title=title)
        titles = [Entity.from_elements(surface=title, value=title, type='title')]
        lf = LF(intent, titles=titles)
        text = template['template']
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

    def inform(self):
        intent = 'inform'
        context_title = self.state.curr_title
        template = self.retrieve_response_template(intent, context_title=context_title)
        if template['context_title'] != context_title:
            return self.template_message('ask')
        lf = LF(intent)
        text = template['template']
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

    def send(self):
        print 'curr title:', self.state.curr_title
        action = self.manager.choose_action(state=self.state)
        if not action:
            action = self.retrieve_action()
        if action in ('inform', 'inform-curr-title'):
            return self.inform()
        elif action == 'inform-new-title':
            return self.inform_title()
        elif action == 'done':
            return self.done()
        else:
            return self.template_message(action)
        raise Exception('Uncaught case')
