__author__ = 'anushabala'
from session import Session
import Queue


class HumanSession(Session):
    def __init__(self, agent):
        super(HumanSession, self).__init__(agent)
        self.outbox = Queue.Queue()
        self.inbox = Queue.Queue()
        self.cached_messages = []

    def send(self):
        if not self.outbox.empty():
            return self.outbox.get()
        return None

    def poll_inbox(self):
        if not self.inbox.empty():
            return self.inbox.get()
        return None

    def receive(self, event):
        self.inbox.put(event)

    def enqueue(self, event):
        self.outbox.put(event)


