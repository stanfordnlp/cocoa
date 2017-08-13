'''
A session abstract class maintains state and can read an event and write an
event.
'''

from src.basic.event import Event

class Session(object):
    def __init__(self, agent, **kwargs):
        self.agent = agent  # 0 or 1 (which player are we?)

    def receive(self, event):
        raise NotImplementedError
    def send(self):
        raise NotImplementedError

    def message(self, text):
        return Event.MessageEvent(self.agent, text)

    # TODO: refactor
    # Mutualfriends
    def select(self, item):
        return Event.SelectionEvent(self.agent, item, None)

    # Negotiation
    def offer(self, offer):
        '''
        offer = {'price': float, 'sides', str}
        '''
        return Event.OfferEvent(self.agent, offer, None)

    def accept(self):
        return Event.AcceptEvent(self.agent, None)
    def reject(self):
        return Event.RejectEvent(self.agent, None)
    def quit(self):
        return Event.QuitEvent(self.agent, None)
