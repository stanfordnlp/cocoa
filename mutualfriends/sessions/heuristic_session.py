from simple_session import SimpleSession
from collections import defaultdict
import random
from cocoa.core.event import Event
import copy
DEBUG = 0

#GREETING = ['hi', 'hello', 'hey']
GREETING = ['hi']

class HeuristicSession(SimpleSession):
    def __init__(self, agent, kb, joint_facts, ask):
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
        self.curr_set = {'items': self.kb.items, 'attrs': set()}
        self.possible_set = {'items': self.kb.items, 'attrs': set()}
        self.prev_possible_set = copy.deepcopy(self.possible_set)

        self.informed_facts = set()
        self.last_selected_item = None

        # Control difficulty
        self.joint_facts = joint_facts
        self.ask_action = ask

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

    def update_informed_facts(self, fact):
        '''
        Record what facts we have communicated so that we don't repeat it.
        '''
        for f in fact:
            self.informed_facts.add(self.fact_hash(f))

    def fact_hash(self, fact):
        return tuple(sorted(fact[0]))

    def filter_fact(self, fact):
        '''
        Filter repeated facts.
        '''
        return [f for f in fact if self.fact_hash(f) not in self.informed_facts]

    def choose_fact(self, subset):
        items, checked_attrs = subset['items'], subset['attrs']
        attr_counts = self.count_attrs(items)
        facts = []

        # Talk about a single attribute
        # Select the attribute with the minimum number of values
        attrs = self.get_majority_attrs(attr_counts, checked_attrs)
        # TODO: if agents list all values in each column, even if all attrs are checked,
        # there can still be no deterministic answer. need to check joint attributes in this case.
        if len(attrs) == 0:
            return None
        attr_name = random.choice(attrs)
        fact = self.attr_facts(attr_counts, attr_name)
        # Talk about a single value
        facts.extend([[f] for f in fact])
        # Talk about all values
        if self.joint_facts:
            if len(fact) > 1:
                facts.append(fact)
        # Always talk about one attribute at the beginning
        if len(self.informed_facts) == 0:
            return random.choice(facts)

        # Talk about joint attributes
        if self.joint_facts:
            # Randomly select two attributes
            attr_names = random.sample(attr_counts.keys(), 2)
            # Randomly select one item to fill in attribute values
            item = random.choice(items)
            attr_values = [(name, item[name]) for name in attr_names]
            count = sum([1 for it in items if self.satisfy(it, (attr_values, 1))])
            fact = [(attr_values, count)]
            facts.append(fact)

        interesting_facts = [fact for fact in facts if self.interesting_fact(fact, subset)]
        selected = self.sample_fact(interesting_facts)
        if selected is None:
            return self.sample_fact(facts)
        return selected

    def sample_fact(self, facts):
        if len(facts) == 0:
            return None
        i = 0
        while i < 100:
            fact = random.choice(facts)
            fact = self.filter_fact(fact)
            if len(fact) > 0:
                return fact
            i += 1
        return None

    def number_to_str(self, count, total):
        if count == 0:
            return 'no'
        elif count == 1:
            #return random.choice(['one', 'only one'])
            return 'one'
        elif count == total:
            return 'all'
        elif count == 2:
            return 'two'
        elif count > 3:
            return 'most'
        else:
            return str(count)

    def fact_to_str(self, fact, item_set, include_count=True):
        fact_str = []
        total = len(item_set['items'])  # Total number of items in the set being considered
        for attrs, count in fact:
            if include_count:
                s = '%s %s' % (self.number_to_str(count, total), ' and '.join([a[1] for a in attrs]))
            else:
                s = ' and '.join([a[1] for a in attrs])
            fact_str.append(s)
        fact_str = ', '.join(fact_str)
        return fact_str

    def inform(self, fact, item_set):
        '''
        - [([(hobby, cooking)], 3)], e.g. I have 3 cooking.
        - [((hobby, cooking), 3), ((hobby, hiking), 2)], e.g. I have 3 cooking and 2 hiking.
        - [([(hobby, cooking), (loc_pref, indoors)], 3)], e.g. I have 3 cooking and indoors.
        - optional: replace count, e.g. 4->most, 5->all
        - negation: count == 0
        '''
        self.state = ('inform', fact)
        self.update_informed_facts(fact)
        fact_str = self.fact_to_str(fact, item_set)
        conditioned = True if len(item_set['items']) < len(self.curr_set['items']) else False
        # Global information
        if not conditioned:
            message = 'I have %s.' % fact_str
        # Local information (conditioned)
        else:
            # TODO: mentioned previous attributes, e.g. The cook like outdoors
            message = 'I have %s in those.' % fact_str
        self.filter(item_set, fact)
        if DEBUG:
            print self.state
        return self.message(message, self.state)

    def reset_possible_set(self):
        self.possible_set['items'] = list(self.curr_set['items'])
        self.possible_set['attrs'] = set(self.curr_set['attrs'])

    def answer(self):
        # TODO: partial satisfaction
        if len(self.possible_set['items']) == 0:
            self.state = ('answer', False)
            self.reset_possible_set()
            return self.message('No.', self.state)
        else:
            self.state = ('answer', True)
            return self.message('Yes.', self.state)

    def ask(self, fact, item_set):
        self.state = ('ask', fact)
        self.update_informed_facts(fact)
        message = 'Do you have %s?' % self.fact_to_str(fact, item_set, False)
        if DEBUG:
            print self.state
        return self.message(message, self.state)

    def select(self, item):
        self.state = ('select', item)
        # Don't repeatedly select one item
        if item == self.last_selected_item:
            return None
        self.last_selected_item = item
        return super(HeuristicSession, self).select(item)

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

    # TODO: attach state in message for now, need to parse state from raw utterance
    def message(self, text, state=None):
        event = Event(agent=self.agent, time=None, action='message', data=text)
        event.state = state
        return event

    def update(self, state):
        #print 'update agent=%d' % self.agent, state
        if state is None:
            return
        # We selected a wrong item
        if self.state and self.state[0] == 'select':
            self.curr_set['items'] = [item for item in self.curr_set['items'] if item != self.state[1]]
            self.possible_set['items'] = [item for item in self.possible_set['items'] if item != self.state[1]]

        self.received_state = state
        intent = state[0]
        facts = None
        certain_facts = []
        if intent in ['ask', 'inform']:
            facts = state[1]
            # The partner's constraint is global (NOTE: assumes facts are disjoint)
            if sum([fact[1] for fact in facts]) == len(self.kb.items):
                certain_facts = facts
        elif intent == 'answer':
            assert self.state[0] in ['ask', 'inform']
            if state[1] is False:
                # Negate all facts
                facts = [(fact[0], 0) for fact in self.state[1]]
                # If this is No to all items, enforce the constraint globally
                if sum([fact[1] for fact in self.state[1]]) == len(self.curr_set['items']):
                    certain_facts = facts
            else:
                facts = self.state[1]

        # Exclude items based on must-have and must-not-have attribute values
        if len(certain_facts) > 0:
            print 'certain facts:', certain_facts
            self.filter(self.curr_set, certain_facts)
            assert len(self.curr_set['items']) > 0

        # Hypothetical items
        if facts:
            if len(self.possible_set['items']) == 0:
                self.reset_possible_set()
            self.filter(self.possible_set, facts)

        if DEBUG:
            print 'agent=%d update:' % self.agent
            print 'possible items:'
            self.print_items(self.possible_set)
            print 'curr items:'
            self.print_items(self.curr_set)

    def interesting_fact(self, fact, subset):
        total = len(subset['items'])
        for attrs, count in fact:
            if count == 1 or count == total:
                return True
        return False

    def receive(self, event):
        self.prev_possible_set = copy.deepcopy(self.possible_set)
        if event.action == 'message':
            # NOTE: assume we can see the state. In practice need to parse.
            self.update(event.state)
        elif event.action == 'select':
            # TODO: update when not matched
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item

    def send(self):
        # We found a match (note that this doesn't always work)
        if self.matched_item:
            return self.select(self.matched_item)

        # Check if we get a match
        if len(self.curr_set['items']) == 1:
            return self.select(self.curr_set['items'][0])
        if len(self.possible_set['items']) == 1 and len(self.possible_set['attrs']) == self.total_num_attrs:
            return self.select(self.possible_set['items'][0])

        #if random.random() < 0.2:  # Wait randomly
        #    return None

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
                return self.answer()

        # Wait for an answer
        if self.state and self.state[0] == 'ask' and (not self.received_state or self.received_state[0] != 'answer'):
            return None

        if len(self.possible_set['items']) == 0:
            self.reset_possible_set()
            # Inform when the partner's constraint results in an empty set
            if self.received_state[0] == 'inform':
                # Negate received positive facts
                fact = [(f[0], 0) for f in self.received_state[1] if f[1] > 0]
                fact = self.filter_fact(fact)
                if len(fact) > 0:
                    return self.inform(fact, self.prev_possible_set)

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
        if self.ask_action and random.random() < 0.5:
            return self.ask(fact, subset)
        else:
            return self.inform(fact, subset)
