import random
import numpy as np
import pdb
from core.event import Event
from sessions.rulebased_session import CraigslistRulebasedSession

class HybridSession(object):
    @classmethod
    def get_session(cls, agent, kb, lexicon, generator, manager, config=None):
        if kb.role == 'buyer':
            return BuyerHybridSession(agent, kb, lexicon, config, generator, manager)
        elif kb.role == 'seller':
            return SellerHybridSession(agent, kb, lexicon, config, generator, manager)
        else:
            raise ValueError('Unknown role: %s', kb.role)

class BaseHybridSession(CraigslistRulebasedSession):
    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        # process the rulebased portion
        utterance = self.parser.parse(event, self.state)
        print('action fed into neural mananger: {}'.format(utterance.lf))
        self.state.update(self.partner, utterance)
        # process the neural based portion
        if event.action == "message":
            logical_form = {"intent": utterance.lf.intent, "price": utterance.lf.price}
            entity_tokens = self.manager.env.preprocessor.lf_to_tokens(self.kb, logical_form)
        else:
            logical_form = None
            entity_tokens = self.manager.env.preprocessor.process_event(event, self.kb)
        if entity_tokens:
            self.manager.dialogue.add_utterance(event.agent, entity_tokens, logical_form)

    # called by the send() method of the parent rulebased session
    def choose_action(self):
        self.manager.dialogue.is_int = False
        action = self.manager.generate()[0]
        print("action predicted by neural manager: {}".format(action))
        if action == "unknown":
            action = "agree" if self.state.partner_act == "accept"
            action = "agree" if self.state.partner_act == "agree"
        if not action:
            action = self.retrieve_action()
            if not action in self.manager.available_actions(self.state):
                action = 'unknown'
        return action

class SellerHybridSession(BaseHybridSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super(SellerHybridSession, self).__init__(agent, kb, lexicon, config, generator, manager)
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

class BuyerHybridSession(BaseHybridSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super(BuyerHybridSession, self).__init__(agent, kb, lexicon, config, generator, manager)
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

