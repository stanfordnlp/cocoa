import random
import numpy as np
from collections import namedtuple
from itertools import izip

from cocoa.core.entity import is_entity
from session import Session
from core.tokenizer import tokenize

class RulebasedSession(object):
    @classmethod
    def get_session(cls, agent, kb, lexicon, config):
        if kb.facts['personal']['Role'] == 'buyer':
            return BuyerRulebasedSession(agent, kb, lexicon, config)
        elif kb.facts['personal']['Role'] == 'seller':
            return SellerRulebasedSession(agent, kb, lexicon, config)
        else:
            raise ValueError('Unknown role: %s', kb['personal']['Role'])

Config = namedtuple('Config', ['overshoot', 'bottomline_fraction', 'compromise_fraction', 'good_deal_threshold'])

default_config = Config(.2, .3, .5, .5)

def random_configs(n=10):
    overshoot = np.random.uniform(0, .5, 10)
    bottomline_fraction = np.random.uniform(.1, .5, 10)
    compromise_fraction = np.random.uniform(.1, .3, 10)
    good_deal_threshold = np.random.uniform(0., 1., 10)
    configs = set([Config(o, b, c, g) for o, b, c, g  in izip(overshoot, bottomline_fraction, compromise_fraction, good_deal_threshold)])
    return list(configs)

class BaseRulebasedSession(Session):
    def __init__(self, agent, kb, lexicon, config):
        super(BaseRulebasedSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.lexicon = lexicon
        if config is None:
            config = default_config

        self.my_price = None
        self.partner_price = None
        self.target = self.kb.facts['personal']['Target']
        self.list_price = self.kb.facts['item']['Price']
        self.category = self.kb.facts['item']['Category']

        # Direction of desired price
        self.inc = None

        self.config = config
        self.overshoot = config.overshoot
        self.bottomline_fraction = config.bottomline_fraction
        self.compromise_fraction = config.compromise_fraction
        # target = 1, bottomline = 0
        self.good_deal_threshold = config.good_deal_threshold

        self.state = {
                'said_hi': False,
                'introduced': False,
                'last_proposed_price': None,
                'num_partner_insist': 0,
                'offered': False,
                'partner_offered': False,
                'final_called': False,
                'num_utterance_sent': 0,
                'last_utterance': None,
                'last_act': None,
                'num_persuade': 0,
                'sides': set(),
                }

        self.persuade_price = []
        self.persuade_detail = []
        self.sides = {}

    @classmethod
    def get_fraction(cls, zero, one, fraction):
        """Return the point at a specific fraction on a line segment.

        Given two points "zero" and "one", return the point in the middle that
        divides the segment by the ratio fraction : (1 - fraction).

        Args:
            zero (float): value at point "zero"
            one (float): value at point "one"
            fraction (float)

        Returns:
            float

        """
        return one * fraction + zero * (1. - fraction)

    @classmethod
    def round_price(cls, price, multiple=10):
        if price > multiple:
            return int(price) / multiple * multiple
        else:
            return price

    def estimate_bottomline(self):
        raise NotImplementedError

    def init_price(self):
        '''
        Initial offer
        '''
        self.bottomline = self.estimate_bottomline()

        # Seller: The target/listing price is shown.
        if self.inc == 1:
            self.my_price = self.target
        else:
            self.my_price = self.target * (1 + self.inc * self.overshoot)

    def receive(self, event):
        if event.action == 'message':
            self.state['num_utterance_sent'] = 0
            raw_utterance = event.data
            entity_tokens = self.lexicon.link_entity(tokenize(raw_utterance), kb=self.kb, scale=False)
            #print 'entity tokens:', entity_tokens
            # Randomly choose one of the prices
            prices = [x[1][0] for x in entity_tokens if is_entity(x)]
            if len(prices) > 0:
                price = prices[-1]
            else:
                # Assuming price is the same as before
                price = self.partner_price
        elif event.action == 'offer':
            self.state['num_utterance_sent'] = 0
            price = event.data['price']
            self.state['partner_offered'] = True
        else:
            return
        if self.partner_price is not None and \
            self.inc*price >= self.inc*self.partner_price:
                self.state['num_partner_insist'] += 1
        elif self.partner_price is None and self.state['last_proposed_price'] is not None:
            self.state['num_partner_insist'] += 1
        else:
            self.state['num_partner_insist'] = 0
        if price is not None:
            self.partner_price = price

    def greet(self):
        greetings = ('hi', 'hello', 'hey')
        self.state['last_act'] = 'greet'
        return self.message(random.choice(greetings))

    def init_propose(self, price):
        raise NotImplementedError

    def propose(self, price):
        # Initial proposal
        if not self.state['last_proposed_price'] and not self.partner_price:
            s = self.init_propose(price)
        else:
            s = (
                    "Can you take %d?" % price,
                    "What about %d?" % price,
                    "What do you think of %d?" % price,
                    "ok, I guess I can take %d" % price,
                    "%d and we have a deal" % price,
                )
        self.state['last_proposed_price'] = price
        self.state['last_act'] = 'propose'
        return self.message(random.choice(s))

    def intro(self):
        raise NotImplementedError

    def _compromise_price(self, price):
        partner_price = self.partner_price if self.partner_price is not None else self.bottomline
        if partner_price is None:
            # TODO: make it a parameter
            my_price = price * (1 - self.inc * 0.1)
        else:
            my_price = self.get_fraction(partner_price, price, self.config.compromise_fraction)
        if self.bottomline is not None and self.compare(my_price, self.bottomline) <= 0:
            return self.bottomline
        else:
            return my_price

    def compromise_price(self):
        if self.bottomline is not None and self.compare(self.my_price, self.bottomline) <= 0:
            return self.final_call()
        self.my_price = self._compromise_price(self.my_price)
        if self.partner_price and self.inc*self.my_price < self.inc*self.partner_price:
            return self.agree()
        # Don't keep compromise
        self.state['num_partner_insist'] = 1  # Reset
        self.state['last_act'] = 'compromise'
        return self.propose(self.my_price)

    def offer_sides(self):
        side_offer = self.sample_templates(self.sides.keys())
        self.state['sides'].add(side_offer)
        return self.message(self.sides[side_offer])

    def compromise(self):
        return self.compromise_price()
        # TODO
        if random.random() < 0.7 or len(self.state['sides']) == len(self.sides):
            return self.compromise_price()
        else:
            return self.offer_sides()

    def persuade(self):
        if self.state['num_persuade'] > 3:
            if not self.state['last_proposed_price']:
                return self.propose(self.my_price)
            return self.compromise()
        if self.state['last_act'] == 'persuade':
            self.state['num_persuade'] += 1
        else:
            self.state['num_persuade'] = 0
        self.state['last_act'] = 'persuade'

        s = self.persuade_detail
        if self.partner_price is not None:
            s.extend(self.persuade_price)
        u = self.sample_templates(s)
        return self.message(u)

    def offer(self, price, sides=''):
        self.state['offered'] = True
        #if not sides and len(self.state['sides']) > 0:
        #    sides = '; '.join(self.state['sides'])
        return super(BaseRulebasedSession, self).offer({'price': price})

    def agree(self):
        self.my_price = self.partner_price
        self.state['last_act'] = 'agree'
        return self.offer(self.my_price)
        # TODO
        if random.random() < 0.5:
            return self.offer(self.my_price)
        else:
            s = (
                    'ok!',
                    'Deal.',
                    'I can take that.',
                    'I can do that.',
                )
            return self.message(random.choice(s))

    def deal(self, price):
        if self.bottomline is None:
            return False
        good_price = self.get_fraction(self.bottomline, self.target, self.config.good_deal_threshold)
        # Seller
        if self.inc == 1 and (
                price >= min(self.list_price, good_price) or \
                price >= self.my_price
                ):
            return True
        # Buyer
        if self.inc == -1 and (
                price <= good_price or\
                price <= self.my_price
                ):
            return True
        return False

    def no_deal(self, price):
        if self.inc*price < self.inc*self.bottomline:
            return True
        return False

    def final_call(self):
        self.state['final_called'] = True

    def sample_templates(self, s):
        for i in xrange(10):
            u = random.choice(s)
            if u != self.state['last_utterance']:
                break
        self.state['last_utterance'] = u
        return u

    def compare(self, x, y):
        """Compare prices x and y.

        For the seller, higher is better; for the buyer, lower is better.

        Args:
            x (float)
            y (float)

        Returns:
            -1: y is a better price
            0: x and y is the same
            1: x is a better price

        """
        raise NotImplementedError

    def send(self):
        if self.bottomline is None:
            self.bottomline = self.estimate_bottomline()

        if self.state['offered']:
            return None

        if self.state['num_utterance_sent'] > 0:
            return None
        self.state['num_utterance_sent'] += 1

        if self.state['partner_offered']:
            if self.no_deal(self.partner_price):
                return self.reject()
            return self.accept()

        if not self.state['said_hi']:
            self.state['said_hi'] = True
            return self.greet()

        if not self.state['introduced'] and self.partner_price is None:
            self.state['introduced'] = True
            return self.intro()

        if self.state['final_called']:
            return self.offer(self.bottomline if self.compare(self.bottomline, self.partner_price) > 0 else self.partner_price)

        if self.state['num_partner_insist'] > 2:
            return self.compromise()

        if self.partner_price is None:
            if self.state['last_proposed_price'] != self.my_price:
                return self.propose(self.my_price)
            else:
                return self.persuade()
        elif self.deal(self.partner_price):
            return self.agree()
        elif self.no_deal(self.partner_price):
            return self.persuade()
        else:
            return self.persuade()
            # TODO
            p = random.random()
            #if p < 0.2:
            #    return self.agree()
            if p < 0.5:
                return self.compromise()
            else:
                return self.persuade()

        raise Exception('Uncatched case')

class SellerRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config):
        super(SellerRulebasedSession, self).__init__(agent, kb, lexicon, config)
        # Direction of desired price
        self.inc = 1.
        self.init_price()

        # Side offers
        self.sides = {
            'credit': "I can accept credit card",
            'extra': "I can throw in a $10 Amazon gift card.",
            }
        if self.category != 'furniture':
            self.sides['delivery'] = "I can deliver it tomorrow"

        if self.category == 'car':
            self.sides.update({
                'warranty': "I can give you one year warranty.",
                'fix scratch': "I can fix the scratches.",
                })
        elif self.category == 'housing':
            self.sides.update({
                'first month free for two-year lease': "If you can sign a two-year lease, I can waive the first month's rent",
                'pets allowed': "Will you have pets? Pets are allowed for you.",
                'new appliance': "I can update some kitchen appliance if needed",
                })
        elif self.category == 'furniture':
            self.sides.update({
                'extra': "I can throw in a few books.",
                })
        elif self.category == 'bike':
            self.sides.update({
                'extra': "It will come with a lock and the headlight.",
                })
        elif self.category == 'phone':
            self.sides.update({
                'extra': "I can throw in a few screen protectors.",
                })

        # Persuade
        self.persuade_price = [
                "This is a steal!",
                "Can you go a little higher?",
                "There is no way I can sell at that price",
                ]
        if self.category == 'car':
            self.persuade_detail = [
                "This car runs pretty well.",
                "It has low milleage for a car of this age",
                "I've been regularly taking it to maintainence",
                ]
        elif self.category == 'housing':
            self.persuade_detail = [
                "It is in a great location with stores and restuarants",
                "The place has been remodeled.",
                "You will be able to enjoy a nice view from the living room window.",
                ]
        elif self.category == 'furniture':
            self.persuade_detail = [
                "It is solid and sturdy",
                "The color matches with most furniture.",
                "It will show your good taste.",
                ]
        elif self.category == 'bike':
            self.persuade_detail = [
                "The color is really attractive.",
                "It's great for commute.",
                "It's been maintained regularly and is in great shape.",
                ]
        if self.category != 'housing':
            self.persuade_detail.extend([
                "It's been rarely used and is kept in great condition!",
                "It's almost brand new.",
                ])

    def estimate_bottomline(self):
        if self.partner_price is None:
            return None
        else:
            return self.get_fraction(self.partner_price, self.list_price, self.config.bottomline_fraction)

    def compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return -1
        else:
            return 1

    def intro(self):
        title = self.kb.facts['item']['Title']
        s = (
                "I have a %s" % title,
                "I'm selling a %s" % title,
            )
        self.state['last_act'] = 'intro'
        return self.message(random.choice(s))

    def init_propose(self, price):
        # We're showing the listing price so no need to propose
        s = (
                "Do you have any question?",
                "It is hard to find such a deal.",
            )
        self.state['last_act'] = 'init_propose'
        return s

    def final_call(self):
        super(SellerRulebasedSession, self).final_call()
        s = (
                "The absolute lowest I can do is %d" % self.bottomline,
                "I cannot go any lower than %d" % self.bottomline,
                "%d or you'll have to go to another place" % self.bottomline,
            )
        self.state['last_act'] = 'final_call'
        return self.message(random.choice(s))

class BuyerRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config):
        super(BuyerRulebasedSession, self).__init__(agent, kb, lexicon, config)
        # Direction of desired price
        self.inc = -1.
        self.init_price()

        # Side offers
        self.sides = {
            'cash': "I can pay cash",
            }
        if self.category == 'car':
            self.sides.update({
                'warranty': "Do you provide warranty?",
                'delivery': "Can you help deliver it to my place?",
                })
        elif self.category == 'housing':
            self.sides.update({
                'first month free for two-year lease': "If I sign a two-year lease, can you waive the first month's rent?",
                'pets allowed': "Is it okay if I have a small dog?",
                'new appliance': "Is it possible to update some kitchen appliance?",
                })
        elif self.category == 'housing':
            self.sides.update({
                'delivery': "Can you help deliver it to my place?",
                })

        # Persuade
        self.persuade_price = [
                "Can you go a little lower?",
                "That's way too expensive!",
                "It looks really nice but way out of my budget...",
                ]
        if self.category == 'car':
            self.persuade_detail = [
                    "This car is pretty old...",
                    "Does it have any accident?",
                    "The mileage is too high; it won't run for too long",
                ]
        elif self.category == 'housing':
            self.persuade_detail = [
                    "What is the lighting condition?",
                    "Is the location good for kids?",
                    "This is a really old property.",
                ]
        elif self.category == 'furniture':
            self.persuade_detail = [
                    "Hmm...The color doesn't really match with my place",
                    "Can it be dissembled?",
                ]
        elif self.category == 'bike':
            self.persuade_detail = [
                    "Can you go lower since I need to purchase locks and headlight?",
                ]
        elif self.category == 'phone':
            self.persuade_detail = [
                    "Is it unlocked?",
                ]
        if self.category != 'housing':
            self.persuade_detail.extend([
                    "It looks really nice; why are you selling it?",
                ])

    def estimate_bottomline(self):
        return self.get_fraction(self.list_price, self.target, self.config.bottomline_fraction)

    def compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return 1
        else:
            return -1

    def intro(self):
        s = (
                "How much are you asking?",
            )
        self.state['last_act'] = 'intro'
        return self.message(random.choice(s))

    def init_propose(self, price):
        s = (
                "I would like to take it for %d" % price,
                "I'm thinking to spend %d" % price,
            )
        self.state['last_act'] = 'init_propose'
        return s

    def final_call(self):
        super(BuyerRulebasedSession, self).final_call()
        s = (
                "The absolute highest I can do is %d" % self.bottomline,
                "I cannot go any higher than %d" % self.bottomline,
                "%d is all I have" % self.bottomline,
            )
        self.state['last_act'] = 'final_call'
        return self.message(random.choice(s))

    def persuade(self):
        p = random.random()
        if p < 0.7 or len(self.sides) == len(self.state['sides']):
            return super(BuyerRulebasedSession, self).persuade()
        else:
            return self.offer_sides()
