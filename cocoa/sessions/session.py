import time
from cocoa.basic.event import Event

class BaseSession(object):
    """An abstarct class for instantiating an agent.

    A session maintains the dialogue state and receive/send dialogue events.

    """
    def __init__(self, agent):
        """Construct a session for an agent.

        Args:
            agent (int): agent id (0 or 1).

        """
        self.agent = agent  # 0 or 1 (which player are we?)

    def receive(self, event):
        """Parse the received event and update the dialogue state.

        Args:
            event (Event)

        """
        raise NotImplementedError

    def send(self):
        """Send an event.

        Returns:
            event (Event)

        """
        raise NotImplementedError

    @classmethod
    def timestamp(cls):
        return str(time.time())

    def message(self, text):
        return Event.MessageEvent(self.agent, text, time=self.timestamp())
