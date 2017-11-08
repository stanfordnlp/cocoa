import random
import copy
import sys
import re
import operator
from collections import namedtuple, defaultdict

from cocoa.core.entity import is_entity
from cocoa.model.parser import LogicalForm

from session import Session
from core.tokenizer import tokenize
from core.split_tracker import SplitTracker
from model.parser import Parser

Config = namedtuple('Config', ['target', 'bottomline', 'patience'])
default_config = Config(8, 5, 10)

class RulebasedSession(Session):
    def __init__(self, agent, kb, lexicon, config):
        super(RulebasedSession, self).__init__(agent)
        self.kb = kb

        self.lexicon = lexicon
        self.tracker = SplitTracker()
        self.parser = Parser(agent, kb, lexicon)

        self.item_values = kb.item_values
        self.item_counts = kb.item_counts
        self.items = kb.item_values.keys()
        self.my_proposal = None
        self.their_item_weights = {k: 1 for k in self.items}
        self.config = default_config if config is None else config

        self.state = {
                'selected': False,
                'proposed': False,
                'act': [None, None],
                'proposal': [None, None],
                'num_utterance': 0,
                'time': 0,
                }

        items = [(item, value, self.item_counts[item]) for item, value in self.item_values.iteritems()]
        # Sort items by value from high to low
        self.sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        self.init_proposal()

    def init_proposal(self):
        """Initialize my proposal before the conversation begins.

        If there are 0-value items, take all non-zero ones. Otherwise, take
        `target` points.

        """
        # Has zero-value item
        if self.sorted_items[-1][1] == 0:
            my_split = {item: count for item, value, count in self.sorted_items if value > 0}
        # All items have values
        else:
            points = 0
            target = self.config.target
            my_split = {item: 0 for item in self.items}
            for item, value, count in self.sorted_items:
                for i in xrange(count):
                    my_split[item] += 1
                    points += value
                    if points >= target:
                        break
                if points >= target:
                    break
        split = {self.agent: my_split, 1 - self.agent: {}}
        self.my_proposal = self.tracker.merge_offers(split, self.item_counts, self.agent)

    def get_points(self, offer):
        return sum([count * self.item_values[item] for item, count in offer.iteritems()])

    def compromise(self, my_offer):
        compromised_offer = copy.deepcopy(my_offer)
        item_scores = [(item, self.item_values[item] / float(self.their_item_weights[item]))
                for item, count in my_offer.iteritems()
                if count > 0 and self.their_item_weights[item] > 0]
        item_scores = sorted(item_scores, key=lambda x: x[1])
        item = item_scores[0][0]
        compromised_offer[item] -= 1
        for item, weight in self.their_item_weights.iteritems():
            if weight < 0:
                compromised_offer[item] = self.kb.item_counts[item]
        return compromised_offer

    def negotiate(self):
        their_offer = self.their_proposal[1 - self.agent]
        my_offer = self.my_proposal[self.agent]

        conflict_items = []
        valuable_items = []
        for item, value, count in self.sorted_items[::-1]:
            if their_offer[item] + my_offer[item] > count:
                conflict_items.append(item)
            elif my_offer[item] < count and self.item_values[item] > 0:
                valuable_items.append(item)

        my_points = self.get_points(my_offer)
        my_points_by_them = self.get_points(self.their_proposal[self.agent])

        if (not conflict_items) or my_points_by_them >= min(self.config.target, my_points):
            self.my_proposal = copy.deepcopy(self.their_proposal)
            return self.clarify()

        compromised_offer = self.compromise(my_offer)

        if self.get_points(compromised_offer) < self.config.bottomline:
            return self.disagree()
        if self.state['num_utterance'] >= self.config.patience:
            return self.final_call()
        else:
            self.my_proposal[self.agent] = compromised_offer
            self.my_proposal = self.tracker.merge_offers(self.my_proposal, self.item_counts, self.agent)
            self.state['act'][self.agent] = LogicalForm('propose')
            return self.propose(self.my_proposal)

    def propose(self, proposal):
        self.state['proposed'] = True
        self.my_proposal = proposal
        templates = [
                "i need {my_offer}, you can have the rest.",
                "how about i get {my_offer}?",
                "i would like {my_offer}.",
                "i'll take {my_offer}",
                "Can you give me {my_offer} and you take the rest?",
                "Can I get {my_offer}?",
                ]
        s = random.choice(templates).format(
                my_offer=self.offer_to_string(proposal[self.agent]),
                their_offer=self.offer_to_string(proposal[1-self.agent]))
        return self.message(s)

    def final_call(self):
        self.state['act'][self.agent] = LogicalForm('final_call')
        templates = [
                "sorry no deal. i really need {my_offer}.",
                "I need {my_offer} or i cannot make the deal",
                ]
        return self.message(
                random.choice(templates).format(
                    my_offer=self.offer_to_string(self.my_proposal[self.agent])
                    )
                )

    def disagree(self):
        self.state['act'][self.agent] = LogicalForm('disagree')
        s = ["You drive a hard bargain here! I really need {my_offer}",
            "Sorry, can't take it. i can give you {their_offer}.",
            "That won't work. I need at least {my_offer}.",
            "Sorry, I do need {my_offer}",
            ]
        return self.message(random.choice(s).format(
            my_offer=self.offer_to_string(self.my_proposal[self.agent]),
            their_offer=self.offer_to_string(self.my_proposal[1-self.agent])
            ))

    def agree(self):
        self.state['act'][self.agent] = LogicalForm('agree')

        s = ["Great deal, thanks!",
          "Yes, that sounds good",
          "Perfect, sounds like we have a deal!",
          "OK, it's a deal"]
        return self.message(random.choice(s))

    def offer_to_string(self, offer):
        items = ['{count} {item}{plural}'.format(item=item, count=count, plural='s' if count > 1 else '')
                for item, count in offer.iteritems() if count > 0]
        if len(items) == 1:
            return 'just {}'.format(items[0])
        elif len(items) == 2:
            return ' and '.join(items)
        else:
            return ', '.join(items[:-1]) + ' and ' + items[-1]

    def clarify(self):
        self.state['act'][self.agent] = LogicalForm('clarify')
        s = [
            "so i get {my_offer}, right?",
            "you mean i get {my_offer}?",
            "so you get {their_offer}?",
            ]
        return self.message(random.choice(s).format(
                my_offer=self.offer_to_string(self.my_proposal[self.agent]),
                their_offer=self.offer_to_string(self.my_proposal[1-self.agent])
                )
            )

    def ask(self):
        self.state['act'][self.agent] = LogicalForm('ask')
        s = [
            "what items do you value highly?",
            "what would you like?",
            "what's your proposal?",
            "what are you willing to give me?",
            ]
        return self.message(random.choice(s))

    @property
    def their_proposal(self):
        return self.state['proposal'][self.partner]

    def receive(self, event):
        lf = self.parser.parse(event, self.state, update_state=True)
        if lf and lf.intent == 'propose':
            for item, count in lf.offer[self.partner].iteritems():
                if count > 0:
                    self.their_item_weights[item] += 1
                else:
                    self.their_item_weights[item] -= 1

    def wait(self):
        return None

    def no_deal(self):
        self.state['act'][self.agent] = LogicalForm('no_deal')
        return self.message('no deal then')

    def select(self, split):
        self.state['act'][self.agent] = LogicalForm('select', split=split)
        return super(RulebasedSession, self).select(split)

    @property
    def my_act(self):
        return self.state['act'][self.agent].intent if self.state['act'][self.agent] else None

    @property
    def partner_act(self):
        return self.state['act'][self.partner].intent if self.state['act'][self.partner] else None

    def send(self):
        if self.my_act == 'select':
            return self.wait()

        self.state['time'] += 1

        if self.partner_act == 'reject':
            return self.reject()

        if self.partner_act == 'select' or self.my_act == 'agree':
            return self.select(self.my_proposal[self.agent])

        # Opening utterance
        if self.state['time'] == 1:
            return self.propose(self.my_proposal)

        # Initial offer
        if not self.state['proposed'] and not self.their_proposal:
            return self.propose(self.my_proposal)

        if self.partner_act == 'agree':
            if self.my_act in ('clarify', 'agree'):
                return self.agree()
            else:
                return self.clarify()
        if not self.their_proposal:
            return self.ask()
        elif self.partner_act in ('propose', 'item', 'disagree'):
            return self.negotiate()
        elif self.my_act == 'disagree':
            return self.final_call()
        elif self.my_act == 'no_deal':
            return self.no_deal()
        else:
            return self.message('deal?')

        raise Exception('Uncaught case')
