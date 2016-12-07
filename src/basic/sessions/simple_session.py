import random
import re
from collections import defaultdict
from src.basic.sample_utils import sample_candidates
from session import Session
from src.model.preprocess import tokenize
from src.model.vocab import is_entity
from src.basic.lexicon import Lexicon
import numpy as np

class SimpleSession(Session):
    '''
    The simple system implements a bot that
    - greets
    - asks or informs about a fact in its KB
    - replies 'no' or 'yes' to partner's repsonse
    - selects and item
    '''

    greetings = ['hi', 'hello', 'hey', 'hiya']

    def __init__(self, agent, kb, lexicon):
        super(SimpleSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.lexicon = lexicon
        self.num_items = len(kb.items)
        self.entity_coords = self.get_entity_coords()
        self.entity_weights = self.weight_entity()
        self.item_weights = [1.] * self.num_items

        # Dialogue state
        self.asked_entities = None
        self.answered = False
        self.said_hi = False
        self.matched_item = None
        self.selected = False

    def get_entity_coords(self):
        '''
        Return a dict of {entity: [row]}
        '''
        entity_coords = defaultdict(list)
        for row, item in enumerate(self.kb.items):
            for col, attr in enumerate(self.kb.attributes):
                entity = item[attr.name]
                entity_coords[entity.lower()].append(row)
        return entity_coords

    def count_entity(self):
        '''
        Return a dict of {entity: count}.
        '''
        entity_counts = defaultdict(int)
        for item in self.kb.items:
            for entity in item.values():
                entity_counts[entity.lower()] += 1
        return entity_counts

    def weight_entity(self):
        '''
        Assign weights to each entity.
        '''
        entity_counts = self.count_entity()
        N = float(self.num_items)
        # Scale counts to [0, 1]
        entity_weights = {entity.lower(): count / N for entity, count in entity_counts.iteritems()}
        return entity_weights

    def choose_fact(self):
        num_entities = np.random.randint(1, 5)
        entities = sample_candidates(self.entity_weights.items())
        return self.entity_to_fact(entities), entities

    def entity_to_fact(self, entities):
        facts = []
        while len(entities) > 0:
            i = 0
            entity = entities[i]
            rowi = self.entity_coords[entity]
            fact = [[entity], len(rowi)]
            for j in xrange(i+1, len(entities)):
                entity2 = entities[j]
                rowj = self.entity_coords[entity2]
                intersect = [r for r in rowi if r in rowj]
                if len(intersect) > 0:
                    fact[0].append(entity2)
                    fact[1] = len(intersect)
                    rowi = intersect
            # Remove converted entities
            entities = [entity for entity in entities if entity not in fact[0]]
            facts.append(fact)
        return facts

    def fact_to_str(self, fact, num_items, include_count=True):
        fact_str = []
        total = num_items
        for attrs, count in fact:
            if include_count:
                s = '%s %s' % (self.number_to_str(count, total), ' and '.join([a for a in attrs]))
            else:
                s = ' and '.join([a for a in attrs])
            fact_str.append(s)
        fact_str = ', '.join(fact_str)
        return fact_str

    def number_to_str(self, count, total):
        if count == 0:
            return 'no'
        elif count == 1:
            return 'one'
        elif count == total:
            return 'all'
        elif count == 2:
            return 'two'
        elif count > 2./3. * total:
            return 'most'
        else:
            return str(count)

    def inform(self, facts):
        fact_str = self.fact_to_str(facts, self.num_items)
        message = 'I have %s.' % fact_str
        return self.message(message)

    def ask(self, facts):
        fact_str = self.fact_to_str(facts, self.num_items, include_count=False)
        message = 'Do you have %s?' % fact_str
        return self.message(message)

    def answer(self, entities):
        fact = self.entity_to_fact(entities)
        return self.inform(fact)

    def sample_item(self):
        item_id = np.argmax(self.item_weights)
        return item_id

    def update_entity_weights(self, entities, delta):
        for entity in entities:
            if entity in self.entity_weights:
                self.entity_weights[entity] += delta

    def update_item_weights(self, entities, delta):
        for i, item in enumerate(self.kb.items):
            values = item.values()
            self.item_weights[i] += delta * len([entity for entity in entities if entity in values])

    def send(self):
        if self.matched_item:
            if not self.selected:
                self.selected = True
                return self.select(self.matched_item)
            else:
                return None

        if random.random() < 0.2:  # Wait randomly
            return None

        # Say hi first
        if not self.said_hi:
            self.said_hi = True
            text = random.choice(self.greetings)
            return self.message(text)

        # Reply to questions
        if self.asked_entities is not None:
            response = self.answer(self.asked_entities)
            self.asked_entities = None
            return response

        # Inform or Ask or Select
        if np.random.random() < 0.7:
            facts, entities = self.choose_fact()
            # Decrease weights of entities mentioned
            self.update_entity_weights(entities, -0.5)
            if np.random.random() < 0.5:
                return self.inform(facts)
            else:
                return self.ask(facts)
        else:
            item_id = self.sample_item()
            # Don't repeatedly select one item
            self.item_weights[item_id] = -100
            return self.select(self.kb.items[item_id])

    def receive(self, event):
        if event.action == 'message':
            raw_utterance = event.data
            entity_tokens = self.lexicon.link_entity(tokenize(raw_utterance))
            entities = [word[1][0] for word in entity_tokens if is_entity(word)]

            if re.search(r'do you|\?', raw_utterance.lower()) is not None:
                self.asked_entities = entities

            # Update item weights
            if len(entities) > 0:
                if len([x for x in entity_tokens if x in ('no', 'none', "don't", 'zero')]) > 0:
                    delta = -1
                else:
                    delta = 1
                self.update_item_weights(entities, delta)

            # Increase weights of entities mentioned by the partner
            self.update_entity_weights(entities, 0.5)

        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
