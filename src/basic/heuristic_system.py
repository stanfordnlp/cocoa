from simple_system import SimpleSystem
from collections import defaultdict
import random

GREETING = ['hi', 'hello', 'hey']
NONE = ['no match', 'nope', 'none']

class HeuristicSystem(SimpleSystem):
    def __init__(self, agent, kb):
        self.agent = agent
        self.kb = kb
        self.total_num_attrs = len(self.kb.items[0])

        # All items start with weight 1 and get decreased
        self.weights = [1] * len(self.kb.items)

        # Dialogue state
        self.said_hi = False
        self.received_state = None
        self.state = None
        self.matched_item = None
        # TODO: put items and attrs into state
        #self.curr_items = self.kb.items
        #self.possible_items = self.curr_items
        # Attributes we've checked for the two sets of items
        #self.curr_items_attrs = set()
        #self.possible_items_attrs = set()

        self.curr_set = {'items': self.kb.items, 'attrs': set()}
        self.possible_set = {'items': self.kb.items, 'attrs': set()}

        self.informed_facts = set()

    def count_attrs(self, items):
        state = defaultdict(lambda : defaultdict(int))
        for item in items:
            for name, value in item.iteritems():
                state[name][value] += 1
        return state

    def get_majority_attrs(self, attr_counts, checked_attrs):
        '''
        Return attributes with minimum number of values.
        '''
        sorted_attrs = sorted(\
            [name for name in attr_counts.keys() if name not in checked_attrs],\
            key=lambda name: len(attr_counts[name]))
        # Add ties
        i = 1
        while i < len(sorted_attrs):
            if sorted_attrs[i][1] == sorted_attrs[0][1]:
                i += 1
            else:
                break
        return sorted_attrs[:i]

    def attr_facts(self, attr_counts, attr_name):
        value_counts = sorted(attr_counts[attr_name].items(), key=lambda (value, count): count, reverse=True)
        # Facts about the selected attributes
        fact = [\
            ([(attr_name, value)], count)\
            for value, count in value_counts\
            ]
        return fact

    def choose_fact(self, subset):
        items, checked_attrs = subset['items'], subset['attrs']
        attr_counts = self.count_attrs(items)
        facts = []

        # Talk about a single attribute
        # Select the attribute with the minimum number of values
        attrs = self.get_majority_attrs(attr_counts, checked_attrs)
        attr_name = random.choice(attrs)
        fact = self.attr_facts(attr_counts, attr_name)
        # Talk about a single value
        facts.append([random.choice(fact)])
        # Talk about all values
        if len(fact) > 1:
            facts.append(fact)

        # Talk about joint attributes
        # Randomly select two attributes
        attr_names = random.sample(attr_counts.keys(), 2)
        # Randomly select one item to fill in attribute values
        item = random.choice(items)
        attr_values = [(name, item[name]) for name in attr_names]
        count = sum([1 for it in items if self.satisfy(it, (attr_values, 1))])
        fact = [(attr_values, count)]
        facts.append(fact)

        i = 0
        while i < 100:
            fact = random.choice(facts)
            new_fact = []
            for f in fact:
                attr_tuple = tuple(f[0])
                if attr_tuple not in self.informed_facts:
                    self.informed_facts.add(attr_tuple)
                    new_fact.append(f)
            if len(new_fact) > 0:
                break
            i += 1
        if len(new_fact) == 0:
            return None
        return fact

    def inform(self, fact):
        '''
        - ((name, value), count), e.g. I have 3 cooking.
        - [((name, value), count)], e.g. I have 3 cooking and 2 hiking.
        - ([(name, value)], count), e.g. I have 3 cooking and indoors.
        - optional: replace count, e.g. 4->most, 5->all
        - negation: count == 0
        '''
        self.state = ('inform', fact)
        return self.message(self.state)

    def reset_possible_set(self):
        self.possible_set['items'] = list(self.curr_set['items'])
        self.possible_set['attrs'] = set(self.curr_set['attrs'])

    def answer(self):
        # TODO: partial satisfaction
        if len(self.possible_set['items']) == 0:
            self.state = ('answer', False)
            self.reset_possible_set()
            return self.message(self.state)
        else:
            self.state = ('answer', True)
            return self.message(self.state)

    def ask(self, fact):
        self.state = ('ask', fact)
        return self.message(self.state)

    def select(self, item):
        self.state = ('select', item)
        return super(HeuristicSystem, self).select(item)

    def satisfy(self, item, fact):
        '''
        If the item satisfy the joint constraint
        specified by attribute (name, value) pairs.
        '''
        satisfy = True
        attr_values, count = fact
        for name, value in attr_values:
            if item[name] != value:
                satisfy = False
                break
        return satisfy == (count > 0)

    def filter(self, item_set, facts):
        items, attrs = item_set['items'], item_set['attrs']
        item_ids = []
        for fact in facts:
            # Attributes that don't exist shouldn't be counted as checked
            if fact[1] > 0:
                for name, value in fact[0]:
                    attrs.add(name)
            for i, item in enumerate(items):
                if self.satisfy(item, fact):
                    item_ids.append(i)
        item_set['items'] = [items[i] for i in set(item_ids)]

    def print_items(self, item_set):
        items, attrs = item_set['items'], item_set['attrs']
        for item in items:
            print item.values()
        print attrs

    def add_checked_attrs(self, facts, checked_attrs):
        for fact in facts:
            for name, value in fact[0]:
                checked_attrs.add(name)

    def update(self, state):
        print 'agent=%d update state:' % self.agent, state
        # We selected a wrong item
        if self.state and self.state[0] == 'select':
            self.curr_set['items'] = [item for item in self.curr_set['items'] if item != self.state[1]]

        self.received_state = state
        intent = state[0]
        facts = None
        exclude_facts = []
        if intent in ['ask', 'inform']:
            facts = state[1]
            exclude_facts = [fact for fact in facts if fact[1] == len(self.curr_set['items'])]
        elif intent == 'answer':
            assert self.state[0] in ['ask', 'inform']
            if state[1] is False:
                # Negate all facts
                facts = [(fact[0], 0) for fact in self.state[1]]
                # If this is No to all items, impose the constraint globally
                if sum([fact[1] for fact in self.state[1]]) == len(self.curr_set['items']):
                    exclude_facts = facts
            else:
                facts = self.state[1]

        # Exclude items based on must-have and must-not-have attribute values
        if len(exclude_facts) > 0:
            print 'exclude facts:', exclude_facts
            self.filter(self.curr_set, exclude_facts)
            #self.filter(self.possible_set, exclude_facts)
            assert len(self.curr_set['items']) > 0

        # Hypothetical items
        if facts:
            if len(self.possible_set['items']) == 0:
                self.reset_possible_set()
            self.filter(self.possible_set, facts)

        print 'agent=%d update:' % self.agent
        print 'possible items:'
        self.print_items(self.possible_set)
        print 'curr items:'
        self.print_items(self.curr_set)

    def receive(self, event):
        if event.action == 'message':
            # NOTE: assume we can see the state. In practice need to parse.
            self.update(event.data)
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item

    def send(self):
        print 'agent=%d SEND' % self.agent
        if random.random() < 0.5:  # Wait randomly
            return None

        # We found a match (note that this doesn't always work)
        if self.matched_item:
            # Don't repeatedly select one item
            if self.state[0] == 'select' and self.state[1] == self.matched_item:
                return None
            return self.select(self.matched_item)

        # Check if we get a match
        if len(self.curr_set['items']) == 1:
            return self.select(self.curr_set['items'][0])
        if len(self.possible_set['items']) == 1 and len(self.possible_set['attrs']) == self.total_num_attrs:
            return self.select(self.possible_set['items'][0])

        # Say hi first
        if not self.said_hi:
            self.said_hi = True
            text = random.choice(GREETING)
            return self.message(text)

        # Reply to partner's utterance
        if self.received_state:
            intent = self.received_state[0]
            if intent == 'ask':
                self.received_state = None
                print 'answer to ask'
                return self.answer()

        # Wait for an answer
        if self.state and self.state[0] == 'ask' and (not self.received_state or self.received_state[0] == 'answer'):
            return None

        if len(self.possible_set['items']) == 0:
            print 'answer to empty item'
            # Negate received facts
            print self.received_state
            fact = [(f[0], 0) for f in self.received_state[1]]
            return self.inform(fact)
        else:
            subset = self.possible_set
        # Take a guess
        if len(subset['items']) < 3:
            if random.random() < 0.5:
                return self.select(random.choice(subset['items']))
        # Select a fact to ask or inform
        fact = self.choose_fact(subset)
        # Run out of facts to inform
        if not fact:
            return self.select(random.choice(subset['items']))
        if random.random() < 0.5:
            return self.ask(fact)
        else:
            return self.inform(fact)
