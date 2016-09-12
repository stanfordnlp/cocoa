import random
import re
from collections import defaultdict
from sample_utils import sorted_candidates
from system import System

class SimpleSystem(System):
    '''
    The simple system implements a bot that
    - greets
    - ask about a single attribute in order of decreasing degree
    - replies 'no' or 'yes' to partner's repsonse
    - selects the friend
    NOTE: this system assumes that names are unique and consistent.
    '''
    def __init__(self, agent, kb):
        super(SimpleSystem, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.sorted_attr = kb.sorted_attr()
        self.curr_attr = 0
        # Dict of attribute value and type (e.g., Google: company)
        self.entities = self.kb.schema.get_entities()

        # Attribute names and their weights (start from 1)
        self.attributes = {attr.name: 1. for attr in self.kb.schema.attributes}
        self.num_attributes = len(self.attributes)

        # Attribute values and their weights (start from 1)
        self.values = defaultdict(lambda : defaultdict(int))
        for item in self.kb.items:
            for attr_name, attr_value in item.iteritems():
                self.values[attr_name][attr_value] = 1.

        self.answered = False
        self.got_answer = False
        self.said_hi = False
        self.last_received_attr = None
        self.last_sent_attr = None

        # All items start with weight 1 and get decreased
        self.weights = [1] * len(self.kb.items)

        # Set when partner mentions the name of a friend (note: assume name only shows up once)
        self.matched_item = None
        self.selected = False

    def choose_attr(self):
        '''
        Ask about an attribute in the order of sorted_attr.
        '''
        while True:
            attr_name, attr_value = self.sorted_attr[self.curr_attr][0]
            if self.values[attr_name][attr_value] > 0:
                break
            self.curr_attr += 1
        return (attr_name, attr_value)

    def choose_attr_item_order(self):
        '''
        Ask about an attribute from the top ranked item.
        '''
        # Choose the highest weight item
        index = sorted(range(len(self.weights)), key=lambda i : -self.weights[i])[0]
        #print 'highest index:', index
        # Choose an attribute and a value to ask
        for attr_name, weight in sorted_candidates(self.attributes.items()):
            value = self.kb.items[index][attr_name]
            #print 'checking %s=%f, %s=%f' % (attr_name, weight, value, self.values[attr_name][value])
            if weight > 0 and self.values[attr_name][value] > 0:
                return (attr_name, value)
        raise Exception('No available attribute.')

    def ask(self, attr):
        name, value = attr
        return 'Do you know %s %s?' % (name, value)

    def answer(self, attr):
        attr_name, attr_value = attr
        for item in self.kb.items:
            if item[attr_name] == attr_value:
                return 'Yes.'
        return 'No.'

    def get_entities(self, tokens):
        '''
        Return the (single) entity in tokens.
        '''
        i = 3  # Skip "Do you know"
        attr_value = None
        attr_type = None
        while i < len(tokens):
            # Find longest phrase (if any) that matches an entity
            for l in range(5, 0, -1):
                phrase = ' '.join(tokens[i:i+l])
                if phrase in self.entities:
                    attr_value = phrase
                    attr_type = self.entities[attr_value]
                    break
            # NOTE: this assumes that only one attribute is asked!
            if attr_value:
                break
            i += 1
        return attr_value, attr_type

    def get_attr_name(self, s):
        '''
        Figure out which attributes this entity belongs to.
        '''
        for name in self.attributes:
            if name in s:
                return name
        return None

    def parse(self, utterance):
        '''
        Inverse function of ask().
        '''
        # Split on punctuation
        tokens = re.findall(r"[\w']+|[.,!?;]", utterance)
        attr_value, attr_type = self.get_entities(tokens)
        if not attr_value:
            return None
        else:
            attr_name = self.get_attr_name(utterance)
            return (attr_name, attr_value)

    def sample(self, items):
        '''
        Return the highest weight one.
        items: [(item, weight), ...]
        '''
        item = sorted_candidates(items)[0]
        if item[1] == 0:
            print items
            print item
        assert item[1] != 0
        return item[0]

    def send(self):
        if random.random() < 0.5:  # Wait randomly
            return None

        # We found a match (note that this doesn't always work)
        if self.matched_item and not self.selected:
            self.selected = True
            return self.select(self.matched_item)

        if sum(self.weights) == 0:
            return None

        # Say hi first
        if not self.said_hi:
            self.said_hi = True
            #text = random.choice(['hi', 'hello'])
            text = 'hi'
            return self.message(text)

        # Reply to questions
        if not self.answered and self.last_received_attr:
            self.answered = True
            return self.message(self.answer(self.last_received_attr))

        # Ask the highest weight attribute
        # Don't udpate_mentioned here because it may not be answered
        # Don't ask again before getting an answer of the previous question
        if self.last_sent_attr and not self.got_answer:
            return None
        else:
            attr = self.choose_attr()
            self.last_sent_attr = attr
            self.got_answer = False
            return self.message(self.ask(attr))

    def update_weights(self, attr, yes):
        '''
        Increment weight if item should have (yes==True) attr_value,
        set to zero otherwise.
        '''
        #print 'update:', attr, yes
        attr_name, attr_value = attr
        for i, item in enumerate(self.kb.items):
            if item[attr_name] == attr_value:
                if yes:
                    self.weights[i] += 1
                    # NOTE: this assumes that names are unique and consistent!
                    if attr_name == 'Name':
                        self.matched_item = item
                else:
                    self.weights[i] = 0

    def update_mentioned(self, attr):
        '''
        Update already mentioned attributes so that they will not be asked again.
        '''
        attr_name, attr_value = attr
        if attr_value in self.values[attr_name]:
            self.values[attr_name][attr_value] = 0
        if sum(self.values[attr_name].values()) == 0:
            self.attributes[attr_name] = 0

    def receive(self, event):
        if event.action == 'message':
            attr = self.parse(event.data)
            if attr:
                self.last_received_attr = attr
                self.update_weights(attr, True)
                self.update_mentioned(attr)
                self.answered = False
            elif 'Yes' in event.data:
                self.update_weights(self.last_sent_attr, True)
                self.update_mentioned(self.last_sent_attr)
                self.got_answer = True
            elif 'No' in event.data:
                self.update_weights(self.last_sent_attr, False)
                self.update_mentioned(self.last_sent_attr)
                self.got_answer = True

            # Additional matching check
            non_zero_items = []
            for i, item in enumerate(self.kb.items):
                if self.weights[i] > 0:
                    non_zero_items.append(item)
            if len(non_zero_items) == 1:
                self.matched_item = non_zero_items[0]
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
