import random
from cocoa.core.entity import is_entity
from session import Session
from core.tokenizer import tokenize
import copy
import sys
import re

class RulebasedSession(object):
    @staticmethod
    def get_session(agent, kb, lexicon=None):
        if kb.facts['Role'] == 'first':
            return FirstRulebasedSession(agent, kb, lexicon)
        elif kb.facts['Role'] == 'second':
            return SecondRulebasedSession(agent, kb, lexicon)
        else:
            raise ValueError('Unknown role: %s', kb.facts['Role'])

class BaseRulebasedSession(Session):
    def __init__(self, agent, kb, lexicon):
        super(BaseRulebasedSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb

        self.item_values = kb.facts['Item_values']
        self.item_counts = kb.facts['Item_counts']
        self.bottomline = 8  # self.kb.facts['personal']['Bottomline']
        self.their_offer = {'made': False, 'book':-1, 'hat':-1, 'ball':-1}
        self.my_proposal = {'made': False, 'book':-1, 'hat':-1, 'ball':-1}
        self.pick_strategy()
        self.init_item_ranking()

        self.state = {
                'said_hi': False,
                'introduced': False,
                'mark_deal_agreed': False,
                'last_act': None,
                'num_utterance_sent': 0,  # has_spoken: False
                'last_utterance': None,
                # 'final_called': False,
                # 'num_partner_insist': 0,
                # 'num_persuade': 0,
                # 'sides': set(),
                }

    def pick_strategy(self):
        valuation = copy.deepcopy(self.item_values)
        # if there is one item valued at 8 or higher, that warrants an obsession
        if max(valuation) >= 8:
            self.strategy = "obsessed"
        elif 0 in valuation:
            zero_location = valuation.index(0)
            valuation.pop(zero_location)
            # if there is another 0 in the valuation, then obsessed on that item
            if 0 in valuation:
                self.strategy = "obsessed"
            else:
                self.strategy = "overvalued"
        # if there are no 0-valued items in the set
        else:
            self.strategy = "balanced"

    def init_item_ranking(self):
        values = copy.deepcopy(self.item_values)
        top_idx = values.index(max(values))
        values[top_idx] = -10
        middle_idx = values.index(max(values))
        values[middle_idx] = -10
        bottom_idx = values.index(max(values))

        items = ['book', 'hat', 'ball']
        self.top_item = items[top_idx]
        self.middle_item = items[middle_idx]
        self.bottom_item = items[bottom_idx]

    def greet(self):
        greetings = ('hi', 'hello', 'hey')
        self.state['last_act'] = 'greet'
        return self.message(random.choice(greetings))

    def receive(self, event):
        if event.action == 'message':
            self.state['num_utterance_sent'] = 0
            # entity_tokens = self.lexicon.link_entity(tokenize(raw_utterance), kb=self.kb, scale=False)
            #print 'entity tokens:', entity_tokens
            self.parse_message(event.data)

        if event.action == 'accept' or event.action == 'offer':
            self.state['num_utterance_sent'] = 0
            self.state['mark_deal_agreed'] = True

    def parse_message(self, raw_utterance):
        tokens = tokenize(raw_utterance)
        for token in tokens:
            try:                # We assume that
                i = int(token)  # if we see numbers in their message,
                                # that means they tried to make an offer
                self.their_offer['made'] = True
            except ValueError:
                continue

        if self.their_offer['made']:
            self.ghetto_price_tracker(tokens)

        regexes = [
            re.compile('(D|d)eal'),
            re.compile('I can (take|do|accept)'),
            re.compile('(S|s)ounds (good|great)'),
        ]

        if any([regex.search(raw_utterance) for regex in regexes]):
            self.state['mark_deal_agreed'] = True

    def ghetto_price_tracker(self, tokens):
        for idx, token in enumerate(tokens):
            if re.match('books?', token):
                book_idx = idx
            if re.match('hats?', token):
                hat_idx = idx
            if re.match('balls?', token):
                ball_idx = idx

        self.their_offer['book'] = int(tokens[book_idx - 1])
        self.their_offer['hat'] = int(tokens[hat_idx - 1])
        self.their_offer['ball'] = int(tokens[ball_idx - 1])

    def init_propose(self, price=None):
        raise NotImplementedError

    def propose(self):
        # if I have not yet made a proposal
        if not self.my_proposal['made']:
            s = [self.init_propose()]
        else:
            self.state['last_act'] = 'propose'
            if self.strategy == 'obsessed':
                self.my_proposal[self.top_item] = 1
                self.my_proposal['made'] = True
                s = "How about I take the " + self.top_item + " and you take the rest?"
            elif self.strategy == 'overvalued':
                self.my_proposal[self.top_item] = 1
                self.my_proposal[self.middle_item] = 1
                self.my_proposal['made'] = True
                s = "What if I get the " + self.top_item + " along with a " + \
                             self. middle_item + "and you take the rest?"
            elif self.strategy == 'balanced':
                s = "They all look good to me, what do you want?"
        return self.message(random.choice(s))

    def intro(self):
        raise NotImplementedError

    def agree(self):
        self.state['last_act'] = 'agree'
        # if random.random() < 0.5:
        #     return self.offer(self.my_price)
        # else:
        self.my_proposal['made'] = self.their_offer['made']
        self.my_proposal['book'] = self.item_counts[0] - self.their_offer['book']
        self.my_proposal['hat'] = self.item_counts[1] - self.their_offer['hat']
        self.my_proposal['ball'] = self.item_counts[2] - self.their_offer['ball']

        s = [   'Ok, that sounds great!',
                'Deal.',
                'I can take that.',
                'I can do that.',
            ]
        return self.message(random.choice(s))

    def sample_templates(self, s):
        for i in xrange(10):
            u = random.choice(s)
            if u != self.state['last_utterance']:
                break
        self.state['last_utterance'] = u
        return u

    def meets_bottomline(self, offer):
        book_total = self.item_values[0] * offer['book']
        hat_total = self.item_values[1] * offer['hat']
        ball_total = self.item_values[2] * offer['ball']
        total_points = book_total + hat_total + ball_total
        return total_points >= self.bottomline

    def persuade(self):
        if self.top_item == 'book':
            self.persuade_detail = [
                "I have always been a book worm.",
                "The books come in a set, so I would want them all.",
                "I'm trying to complete my collection of novels in this series.",
                ]
        elif self.category == 'hat':
            self.persuade_detail = [
                "I need to hide a bald spot with the hat.",
                "People tell me I look great with a hat on.",
                "This hat fits perfectly with my head.",
                ]
        elif self.category == 'ball':
            self.persuade_detail = [
                "I have always loved sports.",
                "I need these for my youth rec league.",
                "You would look great in a hat.",
                ]

    def deal_points(self):
        book_total = self.item_values[0] * self.my_proposal['book']
        hat_total = self.item_values[1] * self.my_proposal['hat']
        ball_total = self.item_values[2] * self.my_proposal['ball']
        deal_points = book_total + hat_total + ball_total
        return deal_points

    def send(self):
        if self.state['num_utterance_sent'] > 0:
            return None
        self.state['num_utterance_sent'] += 1

        if self.state['mark_deal_agreed']:
            # If the deal is marked as agreed, then the system continues even when
            # it is bad. The check on deal_points is more of a unit test, rather
            # than to ensure a good deal, since default points are negative.
            if self.deal_points() > 1:
                outcome = {}
                outcome['deal_points'] = self.deal_points()
                outcome['item_split'] = self.my_proposal

                return self.mark_deal_agreed(data=outcome)
            else:
                return self.reject()

        # if not self.state['said_hi']:
        #     self.state['said_hi'] = True
        #     # We might skip greeting
        #     if random.random() < 0.5:
        #         return self.greet()

        # if not self.state['introduced'] and self.partner_price is None:
        #     self.state['introduced'] = True
            # We might skip intro
            # if random.random() < 0.5:
            #     return self.intro()

        # if self.state['final_called']:
        #     return self.offer(self.bottomline)

        # if self.state['num_partner_insist'] > 2:
        #     return self.compromise()

        if not self.their_offer['made']:
            if self.my_proposal['made']:
                self.persuade()
                return self.persuade_detail
            else:
                return self.propose()
        else:
            if self.meets_bottomline(self.their_offer):
                return self.agree()
            else:
                return self.agree()
                # return self.persuade()

        # elif self.deal(self.partner_price):
        #     return self.agree()
        # elif self.no_deal(self.partner_price):
            # TODO: set persuasion strength
            # return self.persuade()
        # else:
        #     p = random.random()
        #     if p < 0.2:
        #         return self.agree()
        #     elif p < 0.4:
        #         return self.compromise()
        #     else:
        #         return self.persuade()

        raise Exception('Uncaught case')

class FirstRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon):
        super(FirstRulebasedSession, self).__init__(agent, kb, lexicon)

    def valid_proposal(self, offer):
        if offer['book'] > self.item_counts[0]:
            return (False, 'book')
        if offer['hat'] > self.item_counts[1]:
            return (False, 'hat')
        if offer['ball'] > self.item_counts[2]:
            return (False, 'ball')
        else:
            return (True,)

    def intro(self):
        s = [  "So what looks good to you?",
                "Which items do you value highly?"
            ]
        self.state['last_act'] = 'intro'
        return self.message(random.choice(s))

    def p_str(self, item):
        if item == 'top':
            proposal_string = str(self.my_proposal[self.top_item])
        elif item == 'mid':
            proposal_string = str(self.my_proposal[self.middle_item])
        elif item == 'btm':
            proposal_string = str(self.my_proposal[self.bottom_item])
        return proposal_string + " "

    def init_propose(self):
        self.my_proposal[self.top_item] = 1
        self.my_proposal[self.middle_item] = 0
        self.my_proposal[self.bottom_item] = 0
        self.my_proposal['made'] = True

        if self.strategy == 'overvalued':
            test_proposal = copy.deepcopy(self.my_proposal)
            test_proposal[self.middle_item] += 1
            if self.meets_bottomline(test_proposal):
                self.my_proposal[self.middle_item] += 1
            else:
                self.my_proposal[self.middle_item] += 2

        elif self.strategy == 'balanced':
            test_proposal = copy.deepcopy(self.my_proposal)
            test_proposal[self.middle_item] += 1
            if self.meets_bottomline(test_proposal):
                self.my_proposal[self.middle_item] += 1
            else:
                test_proposal[self.bottom_item] += 1
                if self.meets_bottomline(test_proposal):
                    self.my_proposal[self.middle_item] += 1
                    self.my_proposal[self.bottom_item] += 1
                else:
                    test_proposal[self.top_item] += 1
                    if self.valid_proposal(test_proposal)[0]:
                        self.my_proposal[self.top_item] += 1
                        self.my_proposal[self.middle_item] += 1
                        self.my_proposal[self.bottom_item] += 1
                    else:
                        self.my_proposal[self.middle_item] += 2
                        self.my_proposal[self.bottom_item] += 1

        s = "How about " + self.p_str('top') + str(self.top_item) + ", " \
                         + self.p_str('mid') + str(self.middle_item) + " and " \
                         + self.p_str('btm') + str(self.bottom_item) + "."
        self.state['last_act'] = 'propose'
        return s

class SecondRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon):
        super(SecondRulebasedSession, self).__init__(agent, kb, lexicon)

    def intro(self):
        s = "Hi, that " + self.top_item + " looks really nice."
        self.state['last_act'] = 'intro'
        return self.message(s)

    def init_propose(self):
        s = [ "Would you like to have all the " + self.bottom_item + "?",
                "The " + self.top_item + " looks good to me. What about you?"
            ]
        self.state['last_act'] = 'propose'
        return s


#############################

class RuleUtteranceTagger(object):
    '''
    Tag utterances from the rulebased bot, i.e the function that generates the
    utterance.
    NOTE: in future we will add the act/function in the utterance event.
    '''
    greetings = ('hi', 'hello', 'hey')

    propose = [
               r'Can you take',
               r'What about',
               r'What do you think of',
               r'ok, I guess I can take',
               r'and we have a deal',
              ]
