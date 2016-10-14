__author__ = 'anushabala'
from session import Session
import datetime
import random


class TimedSessionWrapper(Session):
    """
    TimedSessionWrapper is a wrapper around a Session class that adds timing logic to the send() function in Session.
    This class can be used to wrap around a session that produces event responses generated using rules (or a model) -
    the wrapper will add a delay to the responses sent by the session in order to simulate human typing/action rates.
    """
    CHAR_RATE = 10
    EPSILON = 1500
    SELECTION_DELAY = 1000

    def __init__(self, agent, session):
        super(TimedSessionWrapper, self).__init__(agent)
        self.session = session
        self.last_message_timestamp = datetime.datetime.now()
        self.queued_event = None

    def receive(self, event):
        self.last_message_timestamp = datetime.datetime.now()
        self.session.receive(event)

    def send(self):
        if self.queued_event is None:
            self.queued_event = self.session.send()

        if self.queued_event.action == 'message':
            delay = float(len(self.queued_event.data)) / self.CHAR_RATE * 1000 + random.uniform(0, self.EPSILON)
        elif self.queued_event.action == 'select':
            delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
        else:
            # unsupported event action type?
            return None

        if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
            return None
        else:
            self.last_message_timestamp = datetime.datetime.now()
            return self.queued_event
