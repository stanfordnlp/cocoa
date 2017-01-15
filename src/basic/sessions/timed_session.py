__author__ = 'anushabala'
from session import Session
import time
import random
from collections import deque


class TimedSessionWrapper(Session):
    """
    TimedSessionWrapper is a wrapper around a Session class that adds timing logic to the send() function in Session.
    This class can be used to wrap around a session that produces event responses generated using rules (or a model) -
    the wrapper will add a delay to the responses sent by the session in order to simulate human typing/action rates.
    """
    CHAR_RATE = 7
    EPSILON = 1.5
    SELECTION_DELAY = 1
    REPEATED_SELECTION_DELAY = 10
    PATIENCE = 2

    def __init__(self, agent, session):
        super(TimedSessionWrapper, self).__init__(agent)
        self.session = session
        self.last_message_timestamp = time.time()
        self.queued_event = deque()
        self.prev_action = None
        self.received = False
        self.num_utterances = 0

    def receive(self, event):
        # join and leave events
        if not (event.action == 'select' or event.action == 'message'):
            return
        self.last_message_timestamp = time.time()
        self.session.receive(event)
        self.received = True
        self.num_utterances = 0
        self.queued_event.clear()

    def send(self):
        if self.num_utterances >= 2:
            return None
        if self.received is False and (self.prev_action == 'select' or \
            self.last_message_timestamp + random.uniform(1, self.PATIENCE) > time.time()):
            return None

	if len(self.queued_event) == 0:
            self.queued_event.append(self.session.send())

	event = self.queued_event[0]
        if event is None:
            return self.queued_event.popleft()
        if event.action == 'message':
            delay = float(len(event.data)) / self.CHAR_RATE + random.uniform(0, self.EPSILON)
        elif event.action == 'select':
            delay = self.SELECTION_DELAY + random.uniform(0, self.EPSILON)
            if self.prev_action == 'select':
                delay += self.REPEATED_SELECTION_DELAY
        else:
            # unsupported event action type?
            return None

        if self.last_message_timestamp + delay > time.time():
            return None
        else:
            event = self.queued_event.popleft()
            self.prev_action = event.action
            self.received = False
            self.num_utterances += 1
            self.last_message_timestamp = time.time()
            event.time = str(self.last_message_timestamp)
            return event
