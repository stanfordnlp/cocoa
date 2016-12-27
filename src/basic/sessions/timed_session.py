__author__ = 'anushabala'
from session import Session
import datetime
import random
from collections import deque


class TimedSessionWrapper(Session):
    """
    TimedSessionWrapper is a wrapper around a Session class that adds timing logic to the send() function in Session.
    This class can be used to wrap around a session that produces event responses generated using rules (or a model) -
    the wrapper will add a delay to the responses sent by the session in order to simulate human typing/action rates.
    """
    CHAR_RATE = 10
    EPSILON = 1500
    SELECTION_DELAY = 1000
    REPEATED_SELECTION_DELAY = 10000

    def __init__(self, agent, session):
        super(TimedSessionWrapper, self).__init__(agent)
        self.session = session
        self.last_message_timestamp = datetime.datetime.now()
        self.queued_event = deque()
        self.prev_action = None

    def receive(self, event):
        self.last_message_timestamp = datetime.datetime.now()
        self.session.receive(event)
        self.queued_event.clear()

    def send(self):
	if len(self.queued_event) == 0:
            self.queued_event.append(self.session.send())

	event = self.queued_event[0]
        if event.action == 'message':
            delay = float(len(event.data)) / self.CHAR_RATE * 1000 + random.uniform(0, self.EPSILON)
        elif event.action == 'select':
            delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
            if self.prev_action == 'select':
                delay += self.REPEATED_SELECTION_DELAY
        else:
            # unsupported event action type?
            return None

        if self.last_message_timestamp + datetime.timedelta(milliseconds=delay) > datetime.datetime.now():
            return None
        else:
            self.last_message_timestamp = datetime.datetime.now()
            event = self.queued_event.popleft()
            self.prev_action = event.action
            return event
