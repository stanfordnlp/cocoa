import random
from src.basic.sessions.session import Session
from src.model.preprocess import tokenize
from src.model.vocab import is_entity

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
        self.target = self.kb['personal']['Target']
        self.bottomline = self.kb['personal']['Bottomline']
        self.range_ = abs(self.target - self.bottomline)

        # Direction of desired price
        self.inc = None
        # TODO: set this to empirical human stat
        self.overshoot = random.choice((0, .1, .2))

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
                }

    def init_price(self):
        '''
        Initial offer
        '''
        self.my_price = self.target + self.inc*(self.overshoot * abs(self.target - self.bottomline))

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
            price = event.data
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

    def _compromise(self, price):
        my_price = price * (1 - .2*self.inc)
        if not self.partner_price:
            return my_price
        else:
            middle_price = int((my_price + self.partner_price) * 0.5)
            if self.inc == 1:
                return max(middle_price, my_price)
            else:
                return min(middle_price, my_price)

    def compromise(self):
        self.my_price = self._compromise(self.my_price)
        if self.partner_price and self.inc*self.my_price < self.inc*self.partner_price:
            return self.agree()
        # Don't keep compromise
        self.state['num_partner_insist'] = 1  # Reset
        if self.inc*self.my_price <= self.inc*self.bottomline:
            return self.final_call()
        self.state['last_act'] = 'compromise'
        return self.propose(self.my_price)

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

    def offer(self, price):
        self.state['offered'] = True
        return super(BaseRulebasedSession, self).offer(price)

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
                price >= self.target - 0.2 * self.range_ or \
                price >= self.my_price
                ):
            return True
        # Buyer
        if self.inc == -1 and (
                price <= self.target + 0.2 * self.range_ or\
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

        if self.state['partner_offered']:
            if not self.no_deal(self.partner_price):
                return self.offer(self.partner_price)
            return self.offer(self.my_price)

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

    def intro(self):
        title = self.kb['item']['Title']
        s = (
                "I have a %s" % title,
                "I'm selling a %s" % title,
            )
        self.state['last_act'] = 'intro'
        return self.message(random.choice(s))

    def init_propose(self, price):
        s = (
                "I'm asking for %d" % price,
                "I'm looking to sell it for %d" % price,
            )
        self.state['last_act'] = 'init_propose'
        return s

    def persuade(self):
        super(SellerRulebasedSession, self).persuade()
        # TODO: depend on price
        s = [
                "This car runs pretty well.",
                "I've been taking good care of it.",
            ]
        if self.partner_price is not None:
            s.extend([
                "This is a steal!",
                "Can you go a little higher?",
                "There is no way I can sell at that price",
                "Go a little higher and we'll talk",
                "That's your best offer??",
                ])
        u = self.sample_templates(s)
        self.state['last_utterance'] = u
        return self.message(u)

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

    def persuade(self):
        super(BuyerRulebasedSession, self).persuade()
        # TODO: depend on price
        s = [
                "I'm on a tight budget.",
                "I'm a poor student...",
                "This is an old car...",
                "Is it in good condition?",
                ]
        if self.partner_price is not None:
            s.extend([
                "Can you go a little lower?",
                "That's way too expensive!",
                "I don't think it worth the price",
                ])
        u = self.sample_templates(s)
        self.state['last_utterance'] = u
        return self.message(u)

    def final_call(self):
        super(BuyerRulebasedSession, self).final_call()
        s = (
                "The absolute highest I can do is %d" % self.bottomline,
                "I cannot go any higher than %d" % self.bottomline,
                "%d is all I have" % self.bottomline,
            )
        self.state['last_act'] = 'final_call'
        return self.message(random.choice(s))
