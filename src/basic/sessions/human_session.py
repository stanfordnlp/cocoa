__author__ = 'anushabala'
from session import Session
import Queue


class HumanSession(Session):
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
        print "Receiving event and adding to inbox for agent %d" % self.agent
        print "Session", self
        self.inbox.append(event)
        print event.to_dict(), [e.to_dict() for e in self.inbox]

    def enqueue(self, event):
        print "Adding event to session outbox for agent %d" % self.agent
        print "Session", self
        self.outbox.append(event)
        print event.to_dict(), [e.to_dict() for e in self.outbox]


