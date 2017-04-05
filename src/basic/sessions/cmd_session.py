from session import Session

class CmdSession(Session):
    def __init__(self, agent, kb):
        super(CmdSession, self).__init__(agent)
        self.kb = kb

    def send(self):
        message = raw_input()
        tokens = message.split()
        if len(tokens) >= 2 and tokens[0] == '<select>':
            return self.select(self.kb.items[int(tokens[1])])
        if len(tokens) >= 2 and tokens[0] == '<offer>':
            return self.offer(int(tokens[1]))
        return self.message(message)

    def receive(self, event):
        print event.data
