import random
import re
import logging
import numpy as np
from collections import namedtuple
from itertools import izip

from cocoa.core.entity import is_entity
from cocoa.core.event import Event
from cocoa.model.parser import LogicalForm as LF
from cocoa.sessions.rulebased_session import RulebasedSession as BaseRulebasedSession

from core.tokenizer import tokenize, detokenize
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

Config = namedtuple('Config', ['overshoot', 'bottomline_fraction', 'compromise_fraction', 'good_deal_threshold'])

default_config = Config(.2, .5, .5, .7)

class CraigslistRulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        parser = Parser(agent, kb, lexicon)
        state = DialogueState(agent, kb)
        super(CraigslistRulebasedSession, self).__init__(agent, kb, parser, generator, manager, state, sample_temperature=10.)

        self.kb = kb
        self.title = self.shorten_title(self.kb.facts['item']['Title'])
        self.config = default_config if config is None else config

        self.target = self.kb.target
        self.bottomline = None
        self.listing_price = self.kb.listing_price
        self.category = self.kb.category

        # Direction of desired price
        self.inc = None

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

    def receive(self, event):
        super(CraigslistRulebasedSession, self).receive(event)
        if self.bottomline is None:
            self.bottomline = self.estimate_bottomline()

    def fill_template(self, template, price=None):
        return template.format(title=self.title, price=(price or ''), listing_price=self.listing_price, partner_price=(self.state.partner_price or ''), my_price=(self.state.my_price or ''))

    def template_message(self, intent, price=None):
        print 'template:', intent, price
        template = self.retrieve_response_template(intent, category=self.kb.category, role=self.kb.role)
        if '{price}' in template['template']:
            price = price or self.state.my_price
        else:
            price = None
        lf = LF(intent, price=price)
        text = self.fill_template(template['template'], price=price)
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

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

        return self.template_message('counter-price')

    def offer(self, price):
        utterance = Utterance(logical_form=LF('offer', price=price))
        self.state.update(self.agent, utterance)
        metadata = self.metadata(utterance)
        return super(BaseRulebasedSession, self).offer({'price': price}, metadata=metadata)

    def accept(self):
        utterance = Utterance(logical_form=LF('accept'))
        self.state.update(self.agent, utterance)
        metadata = self.metadata(utterance)
        return super(BaseRulebasedSession, self).accept(metadata=metadata)

    def reject(self):
        utterance = Utterance(logical_form=LF('reject'))
        self.state.update(self.agent, utterance)
        metadata = self.metadata(utterance)
        return super(BaseRulebasedSession, self).reject(metadata=metadata)

    def agree(self, price):
        self.state.my_price = price
        return self.template_message('agree', price=price)

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
        template = self._final_call_template()
        text = template.format(price=self.bottomline)
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

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

        action = self.choose_action()
        if action == 'unknown':
            action = 'counter-price'

        if action in ('counter-price', 'vague-price'):
            return self.compromise()
        elif action == 'offer':
            return self.offer(self.state.curr_price)
        elif action == 'agree':
            return self.agree(self.state.curr_price)
        else:
            return self.template_message(action)

        raise Exception('Uncatched case')

class SellerRulebasedSession(CraigslistRulebasedSession):
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

    def _final_call_template(self):
        s = (
                "The absolute lowest I can do is {price}",
                "I cannot go any lower than {price}",
                "{price} or you'll have to go to another place",
            )
        return random.choice(s)

class BuyerRulebasedSession(CraigslistRulebasedSession):
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

    def _final_call_template(self):
        s = (
                "The absolute highest I can do is {price}",
                "I cannot go any higher than {price}",
                "{price} is all I have",
            )
        return random.choice(s)
