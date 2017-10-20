from session import Session

class CmdSession(Session):
    def __init__(self, agent, kb):
        super(CmdSession, self).__init__(agent)
        self.kb = kb
        print("End chat using <done>")

    def send(self):
        message = raw_input()
        event = self.parse_input(message)
        return event

    def parse_input(self, message):
        """Parse user input from the command line.

        Args:
            message (str)

        Returns:
            Event

        """
        raw_tokens = message.split()
        tokens = self.remove_nonprintable(raw_tokens)

        print tokens

        if tokens[0] == '<done>':
            return self.done()
        else:
            return self.message(" ".join(tokens))

    def receive(self, event):
        print event.data
