import random

from system import System

class SimpleSystem(System):
    '''
    The simple system implements a bot that
    - greets
    - generates attributes in decreasing order
    - generates 
    - replies 'no' or 'yes' to partner's repsonse
    - selects the friend
    TODO
    '''
    def __init__(self, agent, kb):
        super(SimpleSystem, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.name_attr = self.kb.schema.attributes[0].name

        self.said_hi = False

        # All items start with weight 1 and get decreased
        self.weights = [1] * len(self.kb.items)

        # Set when partner mentions the name of a friend (note: assume name only shows up once)
        self.matched_item = None

    def send(self):
        if random.random() < 0.5:  # Wait randomly
            return None
        if sum(self.weights) == 0:
            return None

        # Say hi first
        if not self.said_hi:
            self.said_hi = True
            text = random.choice(['hi', 'hello'])
            return self.message(text)

        # We found a match (note that this doesn't always work)
        if self.matched_item:
            return self.select(self.matched_item)

        # Ask the highest weight name
        index = sorted(range(len(self.weights)), key=lambda i : -self.weights[i])[0]
        self.weights[index] = 0  # Already mentioned
        name = self.kb.items[index][self.name_attr]
        return self.message('do you know %s?' % name)

    def receive(self, event):
        if event.action == 'message':
            for item in self.kb.items:
                if item[self.name_attr] in event.data:
                    self.matched_item = item
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
