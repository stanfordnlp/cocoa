from src.sessions.session import BaseSession
from src.basic.event import Event

class Session(BaseSession):
    def select(self, item):
        """Select an item from the KB.

        Args:
            item ({attribute_name: attribute_value})

        Returns:
            SelectionEvent

        """
        return Event.SelectionEvent(self.agent, item, time=self.timestamp())
