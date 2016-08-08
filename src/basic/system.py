'''
A system abstract class maintains state and can read an event and write an
event.
'''

from event import Event

class System(object):
    def __init__(self, agent):
        self.agent = agent  # 0 or 1 (which player are we?)

    def receive(self, event):
        raise NotImplementedError
    def send(self):
        raise NotImplementedError

    def message(self, text):
        return Event(agent=self.agent, time=None, action='message', data=text)
    def select(self, index):
        return Event(agent=self.agent, time=None, action='select', data=index)
