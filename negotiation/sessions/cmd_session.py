from session import Session


class CmdSession(Session):
    def __init__(self, agent, kb):
        super(CmdSession, self).__init__(agent)
        self.kb = kb

    def send(self):
        message = raw_input()
        event = self.parse_input(message)
        return event

    def parse_input(self, message):
        """Parse user input from the command line.
        Args:  message (str)
        Returns: Event
        """
        raw_tokens = message.split()
        tokens = self.remove_nonprintable(raw_tokens)

        if len(tokens) >= 2 and tokens[0] == '<offer>':
            return self.offer({'price': int(tokens[1]), 'sides': ''})
        elif tokens[0] == '<accept>':
            return self.accept()
        else:
            return self.message(message)

    def receive(self, event):
        print event.data