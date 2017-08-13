import random
from src.basic.sessions.session import Session
from src.model.negotiation.preprocess import tokenize
from src.basic.entity import is_entity

class RulebasedSession(object):
    @classmethod
    def get_session(cls, agent, kb, lexicon):
        if kb.facts['personal']['Role'] == 'buyer':
            return BuyerRulebasedSession(agent, kb, lexicon)
        elif kb.facts['personal']['Role'] == 'seller':
            return SellerRulebasedSession(agent, kb, lexicon)
        else:
            raise ValueError('Unknown role: %s', kb['personal']['Role'])

class BaseRulebasedSession(Session):
    def __init__(self, agent, kb, lexicon, time_to_ddl=1000):
        super(BaseRulebasedSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb.facts
        self.lexicon = lexicon
        # TODO: consider time
        self.time_to_ddl = time_to_ddl

        self.my_price = None
        self.partner_price = None
        self.bottomline = self.kb['personal']['Bottomline']
        self.target = self.kb['personal']['Target']
        if self.bottomline is None:
            self.bottomline = self.target * 0.8
        self.list_price = self.kb['item']['Price']
        self.category = self.kb['item']['Category']
        assert self.category in ('car', 'housing', 'furniture')

        # Direction of desired price
        self.inc = None
        # TODO: set this to empirical human stat
        self.overshoot = random.choice((.1, .2, .3))

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

    def init_price(self):
        '''
        Initial offer
        '''
        if self.bottomline is not None:
            self.my_price = self.bottomline * (1 + self.inc * self.overshoot)
        elif self.target is not None:
            # Buyer. The target/listing price is shown.
            if self.inc == 1:
                self.my_price = self.target
            else:
                self.my_price = self.target * (1 + self.inc * self.overshoot)

    def receive(self, event):
        self.state['num_utterance_sent'] = 0
        if event.action == 'message':
            raw_utterance = event.data
            entity_tokens = self.lexicon.link_entity(tokenize(raw_utterance), partner_kb=self.kb)
            #print 'entity tokens:', entity_tokens
            # Randomly choose one of the prices
            prices = [x[1][0] for x in entity_tokens if is_entity(x)]
            if len(prices) > 0:
                price = random.choice(prices)
            else:
                # Assuming price is the same as before
                price = self.partner_price
        elif event.action == 'offer':
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
        #print 'partner price:', self.partner_price

    def greet(self):
        greetings = ('hi', 'hello', 'hey')
        self.state['last_act'] = 'greet'
        return self.message(random.choice(greetings))

    def init_propose(self, price):
        raise NotImplementedError

    def propose(self, price):
        # Initial proposal
        if not self.state['last_proposed_price']:
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
        my_price = price * (1 - .2*self.inc)
        if not self.partner_price:
            return my_price
        else:
            middle_price = int((my_price + self.partner_price) * 0.5)
            if self.inc == 1:
                return max(middle_price, my_price)
            else:
                return min(middle_price, my_price)

    def compromise_price(self):
        self.my_price = self._compromise_price(self.my_price)
        if self.partner_price and self.inc*self.my_price < self.inc*self.partner_price:
            return self.agree()
        # Don't keep compromise
        self.state['num_partner_insist'] = 1  # Reset
        if self.inc*self.my_price <= self.inc*self.bottomline:
            return self.final_call()
        self.state['last_act'] = 'compromise'
        return self.propose(self.my_price)

    def offer_sides(self):
        side_offer = self.sample_templates(self.sides.keys())
        self.state['sides'].add(side_offer)
        return self.message(self.sides[side_offer])

    def compromise(self):
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
        if random.random() < 0.5:
            return self.offer(self.my_price)
        else:
            s = (
                    'ok!',
                    'Deal.',
                    'I can take that.',
                )
            return self.message(random.choice(s))

    def deal(self, price):
        # Seller
        if self.inc == 1 and (
                price >= min(self.list_price, self.bottomline * 1.2) or \
                price >= self.my_price
                ):
            return True
        # Buyer
        if self.inc == -1 and (
                price <= self.bottomline * 0.8 or\
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

    def send(self):
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
            # We might skip greeting
            if random.random() < 0.5:
                return self.greet()

        if not self.state['introduced'] and self.partner_price is None:
            self.state['introduced'] = True
            # We might skip intro
            if random.random() < 0.5:
                return self.intro()

        if self.state['final_called']:
            return self.offer(self.bottomline)

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
            # TODO: set persuasion strength
            return self.persuade()
        else:
            p = random.random()
            if p < 0.2:
                return self.agree()
            elif p < 0.4:
                return self.compromise()
            else:
                return self.persuade()

        raise Exception('Uncatched case')

class SellerRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon):
        super(SellerRulebasedSession, self).__init__(agent, kb, lexicon)
        # Direction of desired price
        self.inc = 1.
        self.init_price()

        # Side offers
        self.sides = {
            'credit': "I can accept credit card",
            }
        if self.category == 'car':
            self.sides.update({
                'warranty': "I can give you one year warranty.",
                'fix scratch': "I can fix the scratches.",
                'delivery': "I can deliver it tomorrow",
                })
        elif self.category == 'housing':
            self.sides.update({
                'first month free for two-year lease': "If you can sign a two-year lease, I can waive the first month's rent",
                'pets allowed': "Will you have pets? Pets are allowed for you.",
                'new appliance': "I can update some kitchen appliance if needed",
                })
        elif self.category == 'furniture':
            self.sides.update({
                'delivery': "I can deliver it tomorrow",
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


    def intro(self):
        title = self.kb['item']['Title']
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
    def __init__(self, agent, kb, lexicon):
        super(BuyerRulebasedSession, self).__init__(agent, kb, lexicon)
        # Direction of desired price
        self.inc = -1.
        self.init_price()

        # Side offers
        self.sides = {
            'credit': "Do you accept credit card?",
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
                    "It looks really nice; why are you selling it?",
                ]

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
