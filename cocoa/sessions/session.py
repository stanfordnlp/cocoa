import time
import string
from cocoa.core.event import Event


class Session(object):
    """An abstarct class for instantiating an agent.

    A session maintains the dialogue state and receive/send dialogue events.

    """
    def __init__(self, agent, config=None):
        """Construct a session for an agent.

        Args:
            agent (int): agent id (0 or 1).

        """
        self.agent = agent  # 0 or 1 (which player are we?)
        self.partner = 1 - agent
        self.config = config

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

    @staticmethod
    def remove_nonprintable(raw_tokens):
        tokens = []
        for token in raw_tokens:
            all_valid_characters = True
            for char in token:
                if not char in string.printable:
                    all_valid_characters = False
            if all_valid_characters:
                tokens.append(token)
        return tokens

    @staticmethod
    def timestamp():
        return str(time.time())

    def message(self, text, metadata=None):
        return Event.MessageEvent(self.agent, text, time=self.timestamp(), metadata=metadata)

    def wait(self):
        return None
