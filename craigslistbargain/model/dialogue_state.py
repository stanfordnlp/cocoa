from cocoa.model.dialogue_state import DialogueState as State

class DialogueState(State):
    def __init__(self, agent, kb):
        super(DialogueState, self).__init__(agent, kb)
        self.price = [None, None]
        self.curr_price = None
        self.num_inquiry = 0

    @property
    def listing_price(self):
        return self.kb.listing_price

    @property
    def my_price(self):
        return self.price[self.agent]

    @my_price.setter
    def my_price(self, price):
        self.price[self.agent] = price

    @property
    def partner_price(self):
        return self.price[self.partner]

    def update(self, agent, utterance):
        super(DialogueState, self).update(agent, utterance)
        if not utterance:
            return
        lf = utterance.lf
        if hasattr(lf, 'price') and lf.price is not None:
            self.price[agent] = lf.price
            self.curr_price = lf.price
        if agent == self.agent and lf.intent == 'inquiry':
            self.num_inquiry += 1
