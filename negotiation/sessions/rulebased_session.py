import random
import re
import logging
import numpy as np
from collections import namedtuple
from itertools import izip

from cocoa.core.entity import is_entity
from cocoa.core.event import Event
from cocoa.model.parser import LogicalForm as LF

from core.tokenizer import tokenize, detokenize
from session import Session
from model.parser import Parser, Utterance
from model.dialogue_state import DialogueState

class RulebasedSession(object):
    @classmethod
    def get_session(cls, agent, kb, lexicon, config, generator, manager):
        if kb.role == 'buyer':
            return BuyerRulebasedSession(agent, kb, lexicon, config, generator, manager)
        elif kb.role == 'seller':
            return SellerRulebasedSession(agent, kb, lexicon, config, generator, manager)
        else:
            raise ValueError('Unknown role: %s', kb.role)

Config = namedtuple('Config', ['overshoot', 'bottomline_fraction', 'compromise_fraction', 'good_deal_threshold', 'sample_temperature'])

default_config = Config(.2, .3, .5, .5, 1.)

DEBUG = True

class BaseRulebasedSession(Session):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super(BaseRulebasedSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.title = self.shorten_title(self.kb.facts['item']['Title'])
        self.parser = Parser(agent, kb, lexicon)
        if config is None:
            config = default_config
        self.config = config
        self.generator = generator
        self.manager = manager

        self.target = self.kb.target
        self.bottomline = None
        self.listing_price = self.kb.listing_price
        self.category = self.kb.category

        # Direction of desired price
        self.inc = None

        self.state = DialogueState(agent, kb)
        self.used_templates = set()

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

    def message(self, text, template=None):
        return Event.MessageEvent(self.agent, text, time=self.timestamp(), template=template)

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

    #def tag_utterance(self, utterance):
    #    """Simple NLU by pattern matching.

    #    Args:
    #        utterance (Utterance)

    #    Returns:
    #        (tag,) (tuple)

    #    """
    #    acts = []

    #    if SpeechActAnalyzer.is_question(utterance):
    #        acts.append(SpeechActs.QUESTION)

    #    if SpeechActAnalyzer.is_agreement(utterance):
    #        acts.append(SpeechActs.AGREEMENT)

    #    return acts

    # TODO: price format
    def utterance_to_template(self, utterance, partner=True):
        prev_prices = {self.agent: [self.state.my_price], 1 - self.agent: [self.state.partner_price], 'listing_price': self.listing_price}
        if partner:
            agent_id = 1 - self.agent
        else:
            agent_id = self.agent
        tokens, _ = TemplateExtractor.parse_prices(utterance.tokens, prev_prices, agent_id)
        tokens = TemplateExtractor.parse_title(tokens, self.kb)
        return detokenize(tokens)

    def receive(self, event):
        utterance = self.parser.parse(event, self.state)
        print 'receive:'
        print utterance
        self.state.update(self.partner, utterance)

        if self.bottomline is None:
            self.bottomline = self.estimate_bottomline()

    def fill_template(self, template, price=None):
        return template.format(title=self.title, price=(price or ''), listing_price=self.listing_price, partner_price=(self.state.partner_price or ''), my_price=(self.state.my_price or ''))

    def init_propose(self, price):
        template = self.choose_template('init-price', context_tag=self.state.partner_act)
        s = self.fill_template(template['template'], price)
        s = self.remove_greeting(s)
        lf = LF('init-price', price=price)
        self.state.update(self.agent, Utterance(logical_form=lf))
        return self.message(s, template=template)

    def choose_template(self, tag, context_tag=None, sample=False):
        if context_tag == 'unknown':
            context_tag = None
        template = self.generator.retrieve(self.state.partner_template, category=self.kb.category, role=self.kb.role, tag=tag, context_tag=context_tag, used_templates=self.used_templates, T=self.config.sample_temperature)
        self.used_templates.add(template['id'])
        template = template.to_dict()
        template['source'] = 'rule'
        return template

    def intro(self):
        #self.state['introduced'] = True
        #self.state.my_act = 'intro'
        template = self.choose_template('intro', sample=True)
        if '{price}' in template:
            price = self.state.my_price
            #self.state['curr_price'] = self.state.my_price
        else:
            price = None
        lf = LF('intro', price=price)
        self.state.update(self.agent, Utterance(logical_form=lf))
        s = self.fill_template(template['template'])
        return self.message(s, template=template)

    def _compromise_price(self, price):
        partner_price = self.state.partner_price if self.state.partner_price is not None else self.bottomline
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

    def compromise(self):
        if self.bottomline is not None and self.compare(self.state.my_price, self.bottomline) <= 0:
            return self.final_call()

        self.state.my_price = self._compromise_price(self.state.my_price)
        if self.state.partner_price and self.compare(self.state.my_price, self.state.partner_price) < 0:
            return self.agree(self.state.partner_price)

        #self.state.my_act = 'counter-price'
        #self.state['curr_price'] = self.state.my_price
        lf = LF('counter-price', price=self.state.my_price)
        self.state.update(self.agent, Utterance(logical_form=lf))
        template = self.choose_template('counter-price', context_tag=self.state.partner_act)
        s = self.fill_template(template['template'], self.state.my_price)
        return self.message(s, template=template)

    def offer(self, price):
        #self.state['offered'] = True
        lf = LF('offer', price=price)
        self.state.update(self.agent, Utterance(logical_form=lf))
        return super(BaseRulebasedSession, self).offer({'price': price})

    def agree(self, price):
        #self.state.my_act = 'agree'
        return self.offer(price)

    def deal(self, price):
        if self.bottomline is None:
            return False
        good_price = self.get_fraction(self.bottomline, self.target, self.config.good_deal_threshold)
        # Seller
        if self.inc == 1 and (
                price >= min(self.listing_price, good_price) or \
                price >= self.state.my_price
                ):
            return True
        # Buyer
        if self.inc == -1 and (
                price <= good_price or\
                price <= self.state.my_price
                ):
            return True
        return False

    def no_deal(self, price):
        if self.compare(price, self.state.my_price) >= 0:
            return False
        if self.bottomline is not None:
            if self.compare(price, self.bottomline) < 0 and abs(price - self.bottomline) > 1:
                return True
        else:
            return True
        return False

    def final_call(self):
        lf = LF('final_call', price=self.bottomline)
        self.state.update(self.agent, Utterance(logical_form=lf))
        #self.state['final_called'] = True
        #self.state.my_act = 'final_call'
        #self.state['curr_price'] = self.bottomline

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
        template = self.choose_template('inquiry', context_tag=self.state.partner_act, sample=True)
        s = self.fill_template(template['template'])
        s = self.remove_greeting(s)
        lf = LF('inquiry')
        self.state.update(self.agent, Utterance(logical_form=lf))
        #self.state.my_act = 'inquiry'
        #self.state['num_inquiry'] += 1
        return self.message(s, template=template)

    def has_done(self, intent):
        return intent in self.state.done

    def retrieve(self):
        temp = self.generator.retrieve(self.state.partner_template, category=self.kb.category, role=self.kb.role, context_tag=context_tag)
        #self.logger.debug(temp['template'])
        if '{price}' in temp['template']:
            return self.compromise()
        elif '<offer>' == temp['template']:
            if self.deal(self.state['curr_price']) and self.state.partner_price:
                return self.agree(self.state['curr_price'])
            else:
                return self.compromise()
        else:
            temp = temp.to_dict()
            temp['source'] = 'retrieve'
            self.state.update(self.agent, Utterance(logical_form=LF(temp['tag'])))
            return self.message(self.fill_template(temp['template']), template=temp)

    def send(self):
        if self.has_done('offer'):
            return self.wait()

        if self.state.partner_act == 'offer':
            if self.no_deal(self.state.partner_price):
                return self.reject()
            return self.accept()

        if self.state.partner_price is not None and self.deal(self.state.partner_price):
            return self.agree(self.state.partner_price)

        if self.has_done('final_call'):
            return self.offer(self.bottomline if self.compare(self.bottomline, self.state.partner_price) > 0 else self.state.partner_price)

        action = self.manager.choose_action(state=self.state)

        if action in ('counter-price', 'vague-price'):
            return self.compromise()
        elif action == 'intro':
            return self.intro()
        elif action == 'inquiry':
            return self.inquire()
        elif action == 'init-price':
            return self.init_propose(self.state.my_price)
        elif action == 'offer':
            return self.offer(self.state.curr_price)
        else:
            print action
            return self.retrieve()

        # TODO:
        # longer context, e.g. inquiry multiple times
        # action == unknown?

        #if self.state.my_act == 'agree':
        #    return self.offer(self.state['curr_price'])

        if not self.has_done('intro') and self.state.partner_price is None:
            #self.logger.debug('intro')
            return self.intro()

        if self.kb.role == 'buyer':
            if self.state['num_inquiry'] < 1:
                #self.logger.debug('inquire')
                return self.inquire()

        # TODO: add inform

        # Initial proposal
        if self.state['curr_price'] is None:
            #self.logger.debug('init propose')
            return self.init_propose(self.state.my_price)

        if self.state.partner_act == 'agree':
            #self.logger.debug('agree')
            if self.state['curr_price'] is not None:
                return self.offer(self.state['curr_price'])

        if self.has_done('final_call'):
            return self.offer(self.bottomline if self.compare(self.bottomline, self.state.partner_price) > 0 else self.state.partner_price)

        if self.state.partner_price is not None and self.deal(self.state.partner_price):
            return self.agree(self.state.partner_price)
        elif self.state.partner_act in ('vague-price', 'counter-price', 'init-price'):
            return self.compromise()
        else:
            #self.logger.debug('retrieve')
            temp = self.generator.retrieve(self.state.partner_template, category=self.kb.category, role=self.kb.role, context_tag=self.state.partner_act)
            #self.logger.debug(temp['template'])
            if '{price}' in temp['template']:
                return self.compromise()
            elif '<offer>' == temp['template']:
                if self.deal(self.state['curr_price']) and self.state.partner_price:
                    return self.agree(self.state['curr_price'])
                else:
                    return self.compromise()
            else:
                self.state.my_act = temp['tag']
                temp = temp.to_dict()
                temp['source'] = 'retrieve'
                return self.message(self.fill_template(temp['template']), template=temp)

        raise Exception('Uncatched case')

class SellerRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super(SellerRulebasedSession, self).__init__(agent, kb, lexicon, config, generator, manager)
        # Direction of desired price
        self.inc = 1.
        self.init_price()

    def estimate_bottomline(self):
        if self.state.partner_price is None:
            return None
        else:
            return self.get_fraction(self.state.partner_price, self.listing_price, self.config.bottomline_fraction)

    def init_price(self):
        # Seller: The target/listing price is shown.
        self.state.my_price = self.target

    def compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return -1
        else:
            return 1

    def final_call(self):
        super(SellerRulebasedSession, self).final_call()
        s = (
                "The absolute lowest I can do is %d" % self.bottomline,
                "I cannot go any lower than %d" % self.bottomline,
                "%d or you'll have to go to another place" % self.bottomline,
            )
        return self.message(random.choice(s))

class BuyerRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super(BuyerRulebasedSession, self).__init__(agent, kb, lexicon, config, generator, manager)
        # Direction of desired price
        self.inc = -1.
        self.init_price()

    def estimate_bottomline(self):
        return self.get_fraction(self.listing_price, self.target, self.config.bottomline_fraction)

    def init_price(self):
        self.state.my_price = self.round_price(self.target * (1 + self.inc * self.config.overshoot))

    def compare(self, x, y):
        if x == y:
            return 0
        elif x < y:
            return 1
        else:
            return -1

    def final_call(self):
        super(BuyerRulebasedSession, self).final_call()
        s = (
                "The absolute highest I can do is %d" % self.bottomline,
                "I cannot go any higher than %d" % self.bottomline,
                "%d is all I have" % self.bottomline,
            )
        return self.message(random.choice(s))
