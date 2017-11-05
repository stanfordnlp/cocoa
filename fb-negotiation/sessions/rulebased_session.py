import random
import copy
import sys
import re
import operator
from collections import namedtuple, defaultdict

from cocoa.core.entity import is_entity

from session import Session
from core.tokenizer import tokenize
from core.split_tracker import SplitTracker

Config = namedtuple('Config', ['target', 'bottomline', 'patience'])
default_config = Config(8, 5, 10)

class RulebasedSession(Session):
    def __init__(self, agent, kb, lexicon, config):
        super(RulebasedSession, self).__init__(agent)
        self.kb = kb

        self.lexicon = lexicon
        self.tracker = SplitTracker()

        self.item_values = kb.item_values
        self.item_counts = kb.item_counts
        self.items = kb.item_values.keys()
        self.my_proposal = None
        self.their_proposal = None
        self.their_item_weights = {k: 1 for k in self.items}
        self.config = default_config if config is None else config

        self.state = {
                'selected': False,
                'proposed': False,
                'my_action': None,
                'their_action': None,
                'need_clarify': False,
                'clarified': False,
                'num_utterance': 0,
                'last_offer': None,
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
        item_scores = [(item, self.item_values[item] / float(self.their_item_weights[item])) for item, count in my_offer.iteritems() if count > 0]
        item_scores = sorted(item_scores, key=lambda x: x[1])
        item = item_scores[0][0]
        compromised_offer[item] -= 1
        #print 'current offer:', my_offer
        #print 'item scores:', item_scores
        #print 'compromised offer:', compromised_offer
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
            self.state['my_action'] = 'propose'
            return self.propose(self.my_proposal)

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

    def is_disagree(self, raw_utterance, tokens):
        if 'deal' in tokens and self.is_neg(tokens):
            return True
        return False

    def is_agree(self, raw_utterance, tokens):
        if re.search(r'ok|okay|deal|fine|yes|yeah|good|work|great|perfect', raw_utterance.lower()) and not self.is_neg(tokens):
            return True
        return False

    def has_item(self, tokens):
        for token in tokens:
            if is_entity(token) and token.canonical.type == 'item':
                return True
        return False

    def final_call(self):
        self.state['my_action'] = 'final_call'
        templates = [
                "sorry no deal. i really need {my_offer}.",
                "I need {my_offer} or i cannot make the deal",
                ]
        return self.message(
                random.choice(templates).format(
                    my_offer=self.offer_to_string(self.my_proposal[self.agent])
                    )
                )

    def intro(self):
        self.state['my_action'] = 'intro'

        s = [  "So what looks good to you?",
                "Which items do you value highly?",
                "Hi, what would you like?"
            ]
        return self.message(random.choice(s))

    def disagree(self):
        self.state['my_action'] = 'disagree'
        s = ["You drive a hard bargain here! I really need {my_offer}",
            "Sorry, can't take it. i can give you {their_offer}.",
            "That won't work. I need at least {my_offer}.",
            ]
        return self.message(random.choice(s).format(
            my_offer=self.offer_to_string(self.my_proposal[self.agent]),
            their_offer=self.offer_to_string(self.my_proposal[1-self.agent])
            ))

    def agree(self):
        self.state['my_action'] = 'agree'
        #self.finalize_my_proposal()

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
        self.state['my_action'] = 'clarify'
        self.state['clarified'] = True
        s = [
            "so i get {my_offer}, right?",
            "so i get {my_offer} and you get {their_offer}?",
            ]
        return self.message(random.choice(s).format(
                my_offer=self.offer_to_string(self.my_proposal[self.agent]),
                their_offer=self.offer_to_string(self.my_proposal[1-self.agent])
                )
            )

    def receive(self, event):
        if event.action in ('reject', 'select'):
            self.state['their_action'] = event.action
        elif event.action == 'message':
            self.state['time'] += 1
            tokens = self.lexicon.link_entity(tokenize(event.data))
            split, need_clarify = self.tracker.parse_offer(tokens, 1-self.agent, self.item_counts)
            if split:
                self.their_proposal = split
                self.state['their_action'] = 'propose'
                for item, count in split[1 - self.agent].iteritems():
                    if count > 0:
                        self.their_item_weights[item] += 1
            else:
                if self.is_agree(event.data, tokens):
                    self.state['their_action'] = 'agree'
                elif self.is_disagree(event.data, tokens):
                    self.state['their_action'] = 'disagree'
                elif self.has_item(tokens):
                    self.state['their_action'] = 'item'
                else:
                    self.state['their_action'] = 'unknown'

    def wait(self):
        return None

    def no_deal(self):
        self.state['my_action'] == 'no_deal'
        return self.message('no deal then')

    def send(self):
        if self.state['their_action'] == 'reject':
            return self.reject()

        if self.state['selected']:
            return self.wait()

        if self.state['their_action'] == 'select' or self.state['my_action'] == 'agree':
            self.state['selected'] = True
            return self.select(self.my_proposal[self.agent])

        self.state['time'] += 1

        # Opening utterance
        if self.state['time'] == 1:
            if random.random() < 0.5: # talk a bit by asking a question
                return self.propose(self.my_proposal)
            else:
                return self.intro()             # to hear their side first

        # Initial offer
        if not self.state['proposed'] and not self.their_proposal:
            #print 'propose'
            return self.propose(self.my_proposal)

        if self.state['their_action'] == 'agree':
            if self.state['my_action'] in ('clarify', 'agree'):
                #print 'agree'
                return self.agree()
            else:
                return self.clarify()
        elif not self.their_proposal:
            return self.propose(self.my_proposal)
        elif self.state['their_action'] in ('propose', 'item'):
            #print 'negotiate'
            return self.negotiate()
        elif self.state['my_action'] == 'disagree':
            return self.final_call()
        elif self.state['their_action'] == 'disagree' or self.state['my_action'] == 'no_deal':
            return self.no_deal()
        else:
            return self.message('deal?')

        raise Exception('Uncaught case')
