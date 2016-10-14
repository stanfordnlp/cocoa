__author__ = 'anushabala'
from session import Session


class HumanSession(Session):
    """
    HumanSession represents a single human agent in a dialogue. This class can be used to enqueue messages sent by the
    agent and retrieve messages received from the other agent in the dialogue.
    """
    def __init__(self, agent):
        super(HumanSession, self).__init__(agent)
        self.outbox = []
        self.inbox = []
        self.cached_messages = []
        # todo implement caching to store message history

    def send(self):
        if len(self.outbox) > 0:
            return self.outbox.pop(0)
        return None

    def poll_inbox(self):
        if len(self.inbox) > 0:
            return self.inbox.pop(0)
        return None

    def receive(self, event):
        self.inbox.append(event)

    def enqueue(self, event):
        self.outbox.append(event)


