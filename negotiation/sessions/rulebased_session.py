import random
import re
import numpy as np
from collections import namedtuple
from itertools import izip

from cocoa.core.entity import is_entity

from analysis.dialogue import Utterance
from analysis.speech_acts import SpeechActAnalyzer, SpeechActs
from core.tokenizer import tokenize, detokenize
from session import Session
from systems.templates import TemplateExtractor

class RulebasedSession(object):
    @classmethod
    def get_session(cls, agent, kb, lexicon, config, templates):
        if kb.role == 'buyer':
            return BuyerRulebasedSession(agent, kb, lexicon, config, templates)
        elif kb.role == 'seller':
            return SellerRulebasedSession(agent, kb, lexicon, config, templates)
        else:
            raise ValueError('Unknown role: %s', kb.role)

# resistance: number of partner insistance before making a concession
Config = namedtuple('Config', ['overshoot', 'bottomline_fraction', 'compromise_fraction', 'good_deal_threshold', 'resistance', 'persuade_sides', 'persuade_price', 'sample_temperature'])

default_config = Config(.2, .3, .5, .5, 1, .33, .33, 1.)

class BaseRulebasedSession(Session):
    def __init__(self, agent, kb, lexicon, config, templates):
        super(BaseRulebasedSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.title = self.shorten_title(self.kb.facts['item']['Title'])
        self.lexicon = lexicon
        if config is None:
            config = default_config
        self.templates = templates

        self.my_price = None
        self.partner_price = None
        self.target = self.kb.target
        self.bottomline = None
        self.listing_price = self.kb.listing_price
        self.category = self.kb.category

        # Direction of desired price
        self.inc = None

        self.config = config

        self.state = {
                'time': 0,
                'said_hi': False,
                'introduced': False,
                'curr_price': None,
                'num_partner_insist': 0,
                'offered': False,
                'partner_offered': False,
                'final_called': False,
                'num_utterance_sent': 0,
                'last_utterance': None,
                'last_act': None,
                'my_act': '<start>',
                'partner_act': '<start>',
                'num_persuade': 0,
                'num_inquiry': 0,
                'sides': set(),
                'partner_acts': [],
                }
        self.used_templates = set()
        self.partner_template = '<start>'

        self.persuade_price = []
        self.product_info = []
        self.sides = {}

    def shorten_title(self, title):
        """If the title is too long, shorten it using a generic name for filling in the templates.
        """
        if len(title.split()) > 3:
            if self.kb.category == 'car':
                return 'car'
            elif self.kb.category == 'housing':
                return 'apartment'
            elif self.kb.category == 'phone':
                return 'phone'
            elif self.kb.category == 'bike':
                return 'bike'
            else:
                return 'item'
        else:
            return title

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

    def round_price(self, price):
        """Round the price so that it's not too specific.
        """
        if price == self.target:
            return price
        if price > 100:
            return int(round(price, -1))
        if price > 1000:
            return int(round(price, -2))
        return int(round(price))

    def estimate_bottomline(self):
        raise NotImplementedError

    def init_price(self):
        """Initial offer.
        """
        raise NotImplementedError

    def tag_utterance(self, utterance):
        """Simple NLU by pattern matching.

        Args:
            utterance (Utterance)

        Returns:
            (tag,) (tuple)

        """
        acts = []

        if SpeechActAnalyzer.is_question(utterance):
            acts.append(SpeechActs.QUESTION)

        if SpeechActAnalyzer.is_agreement(utterance):
            acts.append(SpeechActs.AGREEMENT)

        #sentiment = SpeechActAnalyzer.sentiment(utterance)
        #if sentiment == 1:
        #    acts.append(SpeechActs.POS)
        #elif sentiment == -1:
        #    acts.append(SpeechActs.NEG)

        return acts

    def parse_partner_price(self, prices):
        """When the partner mentions multiple prices, decide which one they meant.
        """
        partner_prices = []
        inc = self.inc * -1.
        curr_price = self.state['curr_price']
        for price in prices:
            price = price.canonical.value
            # Assuming one's latest price is always worse than previous ones
            # Seller might propose the listing price but the buyer won't
            if price != curr_price and \
                (self.partner_price is None or inc * price <= inc * self.partner_price) and \
                (self.kb.role == 'seller' or price != self.listing_price):
                    partner_prices.append(price)
        if partner_prices:
            i = np.argmax(inc * np.array(partner_prices))
            return partner_prices[i]
        return None

    # TODO: price format
    def utterance_to_template(self, utterance, partner=True):
        prev_prices = {self.agent: [self.my_price], 1 - self.agent: [self.partner_price], 'listing_price': self.listing_price}
        if partner:
            agent_id = 1 - self.agent
        else:
            agent_id = self.agent
        tokens, _ = TemplateExtractor.parse_prices(utterance.tokens, prev_prices, agent_id)
        tokens = TemplateExtractor.parse_title(tokens, self.kb)
        return detokenize(tokens)

    def receive(self, event):
        # Reset count
        if event.action in ('message', 'offer'):
            self.state['num_utterance_sent'] = 0

        if event.action == 'message':
            self.state['time'] += 1
            utterance = Utterance.from_text(event.data, self.lexicon, self.kb)
            template = self.utterance_to_template(utterance, partner=True)
            self.state['partner_template'] = template
            # Find the (new) proposed price
            price = self.parse_partner_price(utterance.prices)
            # Tag utterance
            acts = TemplateExtractor.parse_utterance(utterance)
            if self.state['time'] == 1 and not 'price' in acts:
                tag = 'intro'
            elif 'price' in acts and self.state['curr_price'] is None:
                tag = 'init-price'
            elif (not 'price' in acts) and 'vague-price' in acts:
                tag = 'vague-price'
            elif price is not None:
                tag = 'counter-price'
            elif acts == ['question']:
                tag = 'inquiry'
            elif not acts and self.state['my_act'] == 'inquiry':
                tag = 'inform'
            elif 'agree' in acts:
                tag = 'agree'
            else:
                tag = 'unknown'

            # Update state
            self.state['partner_act'] = tag
            if price:
                self.state['curr_price'] = price
                self.partner_price = price

        elif event.action == 'offer':
            price = event.data['price']
            self.state['curr_price'] = price
            self.state['partner_offered'] = True
            self.partner_price = price
        # Decorative events
        else:
            return

        if self.bottomline is None:
            self.bottomline = self.estimate_bottomline()

    def greet(self):
        greetings = ('hi', 'hello', 'hey')
        self.state['last_act'] = 'greet'
        self.state['said_hi'] = True
        return self.message(random.choice(greetings))

    def fill_template(self, template, price=None):
        return template.format(title=self.title, price=(price or ''), listing_price=self.listing_price, partner_price=(self.partner_price or ''), my_price=(self.my_price or ''))

    def init_propose(self, price):
        s = self.fill_template(self.choose_template('init-price', context_tag=self.state['partner_act']), price)
        s = self.remove_greeting(s)
        self.state['curr_price'] = price
        self.my_price = price
        self.state['my_act'] = 'init-price'
        return self.message(s)

    def propose(self, price):
        price = self.round_price(price)
        # Initial proposal
        if not self.state['curr_price']:
            s = self.init_propose(price)
        else:
            s = (
                    "Can you take ${}".format(price),
                    "What about ${}?".format(price),
                    "What do you think of ${}?".format(price),
                    "I guess I can do ${}".format(price),
                    "${} and we have a deal".format(price),
                )
        self.state['curr_price'] = price
        self.state['last_act'] = 'propose'
        self.my_price = price
        msg = random.choice(s)
        return self.message(msg)

    def choose_template(self, response_tag, context_tag=None, sample=False):
        if sample:
            template = self.templates.choose(category=self.kb.category, role=self.kb.role, response_tag=response_tag, context_tag=context_tag, T=self.config.sample_temperature, used_templates=self.used_templates)
        else:
            template = self.templates.search(self.partner_template, category=self.kb.category, role=self.kb.role, response_tag=response_tag, context_tag=context_tag, used_templates=self.used_templates)
        self.used_templates.add(template['id'])
        return template['response']

    def intro(self):
        self.state['introduced'] = True
        self.state['my_act'] = 'intro'
        template = self.fill_template(self.choose_template('intro', sample=True))
        if '{price}' in template:
            self.state['curr_price'] = self.my_price
        s = template.format(title=self.title.encode('utf-8'), price=self.my_price)
        return self.message(s)

    def _compromise_price(self, price):
        partner_price = self.partner_price if self.partner_price is not None else self.bottomline
        if partner_price is None:
            # TODO: make it a parameter
            my_price = price * (1 - self.inc * 0.1)
        else:
            my_price = self.get_fraction(partner_price, price, self.config.compromise_fraction)
        my_price = self.round_price(my_price)
        # Don't go below bottomline
        if self.bottomline is not None and self.compare(my_price, self.bottomline) <= 0:
            return self.bottomline
        else:
            return my_price

    def offer_sides(self):
        side_offer = self.sample_templates(self.sides.keys())
        self.state['sides'].add(side_offer)
        return self.message(self.sides[side_offer].format(price=self.my_price))

    def compromise(self):
        if self.bottomline is not None and self.compare(self.my_price, self.bottomline) <= 0:
            return self.final_call()

        self.my_price = self._compromise_price(self.my_price)
        if self.partner_price and self.compare(self.my_price, self.partner_price) < 0:
            return self.agree(self.partner_price)

        self.state['num_partner_insist'] = 0  # Reset
        self.state['last_act'] = 'compromise'
        self.state['my_act'] = 'counter-price'

        if self.templates is not None:
            self.state['curr_price'] = self.my_price
            s = self.fill_template(self.choose_template('counter-price', context_tag=self.state['partner_act']), self.my_price)
            return self.message(s)
        else:
            return self.propose(self.my_price)

    def persuade(self):
        # Reset
        if self.state['last_act'] != 'persuade':
            self.state['num_persuade'] = 0
        self.state['last_act'] = 'persuade'
        self.state['my_act'] = 'persuade'
        self.state['num_persuade'] += 1

        if self.templates is not None:
            s = self.fill_template(self.choose_template('vague-price', context_tag=self.state['partner_act']))
            return self.message(s)
        else:
            p = random.random()
            if p < self.config.persuade_sides:
                # There are still side offers we haven't mentioned
                if len(self.sides) < len(self.state['sides']):
                    return self.offer_sides()
                else:
                    p += self.config.persuade_sides
            if p < self.config.persuade_price + self.config.persuade_sides:
                if SpeechActs.PRICE in self.state['partner_acts']:
                    return self.complain_price()
            return self.persuade_product()

    def complain_price(self):
        s = self.persuade_price
        u = self.sample_templates(s)
        return self.message(u)

    def persuade_product(self):
        s = self.product_info
        u = self.sample_templates(s)
        return self.message(u)

    def offer(self, price):
        self.state['offered'] = True
        return super(BaseRulebasedSession, self).offer({'price': price})

    def agree(self, price):
        self.state['last_act'] = 'agree'
        return self.offer(price)

    def deal(self, price):
        if self.bottomline is None:
            return False
        good_price = self.get_fraction(self.bottomline, self.target, self.config.good_deal_threshold)
        # Seller
        if self.inc == 1 and (
                price >= min(self.listing_price, good_price) or \
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
        if self.compare(price, self.my_price) >= 0:
            return False
        if self.bottomline is not None:
            if self.compare(price, self.bottomline) < 0 and abs(price - self.bottomline) > 1:
                return True
        else:
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

    def wait(self):
        return None

    # TODO: move this to templates
    def remove_greeting(self, s):
        s = re.sub(r'(hi|hello|hey)[ ,!]*', '', s)
        return s

    def inquire(self):
        s = self.fill_template(self.choose_template('inquiry', context_tag=self.state['partner_act'], sample=True))
        s = self.remove_greeting(s)
        self.state['my_act'] = 'inquiry'
        self.state['num_inquiry'] += 1
        return self.message(s)

    def send(self):
        # Strict turn-taking
        if self.state['num_utterance_sent'] > 0:
            return self.wait()
        self.state['num_utterance_sent'] += 1

        if self.state['offered']:
            return self.wait()

        if self.state['partner_offered']:
            if self.state['my_act'] == 'agree':
                return self.accept()
            if self.no_deal(self.partner_price):
                return self.reject()
            return self.accept()

        if self.state['my_act'] == 'agree':
            return self.offer(self.state['curr_price'])

        self.state['time'] += 1

        if not self.state['introduced'] and self.partner_price is None:
            #print 'INTRO'
            return self.intro()

        if self.kb.role == 'buyer':
            if self.state['num_inquiry'] < 1:
                #print 'INQUIRE'
                return self.inquire()

        # TODO: add inform

        # Initial proposal
        if self.state['curr_price'] is None:
            #print 'INIT PROPOSE'
            return self.init_propose(self.my_price)

        # TODO: check against agree templates
        if self.state['partner_act'] == 'agree':
            #print 'AGREE'
            if self.state['curr_price'] is not None:
                return self.offer(self.state['curr_price'])

        if self.state['final_called']:
            return self.offer(self.bottomline if self.compare(self.bottomline, self.partner_price) > 0 else self.partner_price)

        #if self.state['num_persuade'] > self.config.resistance:
        #    self.state['num_persuade'] = 0
        #    return self.compromise()

        if self.partner_price is not None and self.deal(self.partner_price):
            return self.agree(self.partner_price)
        elif self.state['partner_act'] in ('vague-price', 'counter-price'):
            #print 'COMPROMISE'
            return self.compromise()
        else:
            temp = self.templates.search(self.state['partner_template'], category=self.kb.category, role=self.kb.role)
            if '{price}' in temp['response']:
                return self.compromise()
            else:
                self.state['my_act'] = temp['response_tag']
                return self.message(self.fill_template(temp['response']))

        #if self.partner_price is None:
        #    return self.persuade()
        #elif self.deal(self.partner_price):
        #    return self.agree(self.partner_price)
        #else:
        #    return self.compromise()

        raise Exception('Uncatched case')

class SellerRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, templates):
        super(SellerRulebasedSession, self).__init__(agent, kb, lexicon, config, templates)
        # Direction of desired price
        self.inc = 1.
        self.init_price()

        # Side offers
        self.sides = {
            'credit': "I can accept credit card if you can do {price}.",
            'extra': "I can throw in a $10 Amazon gift card. How does that sound?",
            }

        if self.category != 'housing':
            self.sides['delivery'] = "Can you go higher if I deliver it to you tomorrow?"

        if self.category == 'car':
            self.sides.update({
                'warranty': "I can give you one year warranty for ${price}.",
                'fix scratch': "Could you go higher if I fix the scratches?",
                })
        elif self.category == 'housing':
            self.sides.update({
                'first month free for two-year lease': "If you can sign a two-year lease, I can waive the first month's rent for you.",
                'pets allowed': "Will you have pets? Pets are allowed for you!",
                'new appliance': "I can update some kitchen appliance if needed.",
                })
        elif self.category == 'bike':
            self.sides.update({
                'bike-extra': "I can throw in a lock and my headlight.",
                })
        elif self.category == 'phone':
            self.sides.update({
                'phone-extra': "I can throw in a few screen protectors.",
                })

        # Persuade
        self.persuade_price = [
                "This is a steal!",
                "Can you go a little higher?",
                "There is no way I can sell at that price",
                ]
        if self.category == 'car':
            self.product_info = [
                "This car runs pretty well.",
                "It has low milleage for a car of this age. My price is very fair.",
                "I've been regularly taking it to maintainence.",
                ]
        elif self.category == 'housing':
            self.product_info = [
                "It is in a great location with stores and restuarants. You will not regret.",
                "The place has been remodeled. I can assure you the money is well spent.",
                "You will be able to enjoy a nice view from the living room window.",
                ]
        elif self.category == 'furniture':
            self.product_info = [
                "It is solid and sturdy. Definitely worth the price.",
                "The color matches with most furniture.",
                "It will show your good taste. Definitely worth the price.",
                ]
        elif self.category == 'bike':
            self.product_info = [
                "The color is really attractive. I'm sure you'll like it.",
                "It's great for commute and will save you gas money.",
                "It's been maintained regularly and is in great shape. My price is very fair.",
                ]
        if self.category != 'housing':
            self.product_info.extend([
                "It's been rarely used and is kept in great condition!",
                "It's almost brand new.",
                ])

    def estimate_bottomline(self):
        if self.partner_price is None:
            return None
        else:
            return self.get_fraction(self.partner_price, self.listing_price, self.config.bottomline_fraction)

    def init_price(self):
        # Seller: The target/listing price is shown.
        self.my_price = self.target

    def compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return -1
        else:
            return 1

    def intro(self):
        if self.templates is not None:
            return super(SellerRulebasedSession, self).intro()
        else:
            title = self.kb.facts['item']['Title']
            s = (
                    "I have a %s." % title,
                    "I'm selling a %s." % title,
                    "Are you interested in my %s?" % title,
                )
            return self.message(random.choice(s))

    def init_propose(self, price):
        if self.templates:
            return super(SellerRulebasedSession, self).init_propose(price)
        else:
            # We're showing the listing price so no need to propose
            s = (
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
        self.state['curr_price'] = self.bottomline
        return self.message(random.choice(s))

class BuyerRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, templates):
        super(BuyerRulebasedSession, self).__init__(agent, kb, lexicon, config, templates)
        # Direction of desired price
        self.inc = -1.
        self.init_price()

        # Side offers
        self.sides = {
            'cash': "Can you go lower if I pay cash?",
            }
        if self.category != 'housing':
            self.sides['delivery'] = "What if I come pick it up?"
        elif self.category == 'housing':
            self.sides.update({
                'first month free for two-year lease': "If I sign a two-year lease, can you go lower?",
                'good tenant': "I do not smoke nor do I have pets. I don't think I'll have overnight guests either. Can you consider it?",
                })

        # Persuade
        self.persuade_price = [
                "Can you go a little lower?",
                "That's way too expensive!",
                "It looks really nice but this is way out of my budget...",
                ]
        if self.category == 'car':
            self.product_info = [
                    "This car is pretty old... it's a bit over-priced.",
                    "Does it have any accidents?",
                    "The mileage is too high; it won't run for too long.",
                ]
        elif self.category == 'housing':
            self.product_info = [
                    "What is the lighting condition?",
                    "Is the location good for kids?",
                    "This seems to be a really old property and I'll need to fix things...",
                ]
        elif self.category == 'furniture':
            self.product_info = [
                    "Hmm...The color doesn't really match with my place",
                    "Can it be dissembled?",
                ]
        elif self.category == 'bike':
            self.product_info = [
                    "Can you go lower since I need to purchase locks and headlight?",
                ]
        elif self.category == 'phone':
            self.product_info = [
                    "Is it unlocked?",
                ]
        if self.category != 'housing':
            self.product_info.extend([
                    "It looks really nice; why are you selling it?",
                ])

    def estimate_bottomline(self):
        return self.get_fraction(self.listing_price, self.target, self.config.bottomline_fraction)

    def init_price(self):
        self.my_price = self.round_price(self.target * (1 + self.inc * self.config.overshoot))

    def compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return 1
        else:
            return -1

    def intro(self):
        if self.templates is not None:
            return super(BuyerRulebasedSession, self).intro()
        else:
            title = self.kb.facts['item']['Title']
            s = (
                    "How much are you asking?",
                    "It looks great! Is it still available?",
                    "I'm interested in your {}.".format(title.encode('utf-8')),
                )
            return self.message(random.choice(s))

    def init_propose(self, price):
        if self.templates:
            return super(BuyerRulebasedSession, self).init_propose(price)
        else:
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
