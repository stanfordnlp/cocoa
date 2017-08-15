'''
A session abstract class maintains state and can read an event and write an
event.
'''

import time
from src.basic.event import Event

class Session(object):
    def __init__(self, agent, **kwargs):
        self.agent = agent  # 0 or 1 (which player are we?)

    def receive(self, event):
        raise NotImplementedError
    def send(self):
        raise NotImplementedError

    @classmethod
    def timestamp(cls):
        return str(time.time())

    def message(self, text):
        return Event.MessageEvent(self.agent, text, time=self.timestamp())

    # TODO: refactor
    # Mutualfriends
    def select(self, item):
        return Event.SelectionEvent(self.agent, item, time=self.timestamp())

    # Negotiation
    def offer(self, offer):
        '''
        offer = {'price': float, 'sides', str}
        '''
        return Event.OfferEvent(self.agent, offer, time=self.timestamp())

    def accept(self):
        return Event.AcceptEvent(self.agent, time=self.timestamp())
    def reject(self):
        return Event.RejectEvent(self.agent, time=self.timestamp())
    def quit(self):
        return Event.QuitEvent(self.agent, time=self.timestamp())
