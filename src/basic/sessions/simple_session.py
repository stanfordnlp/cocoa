import random
import re
from collections import defaultdict
from src.basic.sample_utils import sample_candidates
from session import Session
from src.model.preprocess import tokenize, word_to_num
from src.model.vocab import is_entity
from src.basic.lexicon import Lexicon
import numpy as np
from itertools import izip

num_to_word = {v: k for k, v in word_to_num.iteritems()}

class SimpleSession(Session):
    '''
    The simple system implements a bot that
    - greets
    - asks or informs about a fact in its KB
    - replies 'no' or 'yes' to partner's repsonse
    - selects and item
    '''

    greetings = ['hi', 'hello', 'hey', 'hiya']

    def __init__(self, agent, kb, lexicon, realizer=None, consecutive_entity=True):
        super(SimpleSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.attr_type = {attr.name: attr.value_type for attr in kb.attributes}
        self.lexicon = lexicon
        self.realizer = realizer
        self.consecutive_entity = consecutive_entity
        self.num_items = len(kb.items)
        self.entity_coords = self.get_entity_coords()
        self.entity_weights = self.weight_entity()
        self.item_weights = [1.] * self.num_items

        self.sent_entity = False
        self.mentioned_entities = set()

        # Dialogue state
        self.asked_entities = None
        self.answered = False
        self.said_hi = False
        self.matched_item = None
        self.selected = False

        self.capitalize = random.choice([True, False])
        self.numerical = random.choice([True, False])

    def get_entity_coords(self):
        '''
        Return a dict of {entity: [row]}
        '''
        entity_coords = defaultdict(list)
        for row, item in enumerate(self.kb.items):
            for col, attr in enumerate(self.kb.attributes):
                entity = (item[attr.name].lower(), attr.value_type)
                entity_coords[entity].append(row)
        return entity_coords

    def get_related_entity(self, entities):
        '''
        Return entities in the same row and col as the input entities.
        '''
        rows = set()
        types = set()
        for entity in entities:
            rows.update(self.entity_coords[entity])
            types.add(entity[1])
        cols = []
        for i, attr in enumerate(self.kb.attributes):
            if attr.value_type in types:
                cols.append(i)
        row_entities = set()
        col_entities = set()
        for row, item in enumerate(self.kb.items):
            for col, attr in enumerate(self.kb.attributes):
                entity = (item[attr.name].lower(), attr.value_type)
                if entity in entities:
                    continue
                if row in rows:
                    row_entities.add(entity)
                if col in cols:
                    col_entities.add(entity)
        return row_entities, col_entities

    def count_entity(self):
        '''
        Return a dict of {entity: count}.
        '''
        entity_counts = defaultdict(int)
        for item in self.kb.items:
            for attr_name, entity in item.iteritems():
                entity = (entity.lower(), self.attr_type[attr_name])
                entity_counts[entity] += 1
        return entity_counts

    def weight_entity(self):
        '''
        Assign weights to each entity.
        '''
        entity_counts = self.count_entity()
        N = float(self.num_items)
        # Scale counts to [0, 1]
        entity_weights = {entity: count / N for entity, count in entity_counts.iteritems()}
        return entity_weights

    def choose_fact(self):
        num_entities = np.random.randint(1, 3)
        entities = sample_candidates(self.entity_weights.items(), num_entities)
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

    def fact_to_str(self, fact, num_items, include_count=True, prefix=False, question=False):
        fact_str = []
        total = num_items
        for entities, count in fact:
            # Add caninical form as surface form
            entities = [(x[0], x) for x in entities]
            if self.realizer:
                entities_str = self.realizer.realize_entity(entities)
            else:
                entities_str = [x[0] for x in entities]
            if prefix:
                new_str = []
                for entity, s in izip(entities, entities_str):
                    entity_type = entity[1][1]
                    if entity_type == 'name':
                        p = 'named'
                    elif entity_type == 'school':
                        p = 'who went to'
                    elif entity_type == 'company':
                        p = 'working at'
                    elif entity_type == 'major':
                        p = 'who studied'
                    else:
                        p = random.choice(['who like' if count > 1 else 'who like', 'into'])
                    new_str.append(p + ' ' + s)
                entities_str = new_str
            conj = '%s' % ('friends' if count > 1 else 'friend') if prefix else ''
            if include_count:
                s = '%s %s %s' % (self.number_to_str(count, total), conj, ' and '.join([a for a in entities_str]))
            else:
                s = '%s %s' % (conj, ' and '.join([a for a in entities_str]))
            fact_str.append(s)
        if question and len(fact_str) > 1:
            fact_str = ', '.join(fact_str[:-1]) + ' or ' + fact_str[-1]
        else:
            fact_str = ', '.join(fact_str)
        fact_str = fact_str.replace('  ', ' ')
        return fact_str

    def number_to_str(self, count, total):
        if count == 0:
            return 'no'
        elif count == 1:
            return '1'
        elif count == total:
            return 'all'
        elif count == 2:
            return '2'
        elif count > 2./3. * total:
            return 'many'
        else:
            return 'some'

    def naturalize(self, text):
        tokens = text.split()
        if self.capitalize:
            tokens[0] = tokens[0].title()
            tokens = ['I' if x == 'i' else x for x in tokens]
        if not self.numerical:
            tokens = [num_to_word[x] if x in num_to_word else x for x in tokens]
        return ' '.join(tokens)

    def inform(self, facts):
        fact_str = self.fact_to_str(facts, self.num_items, prefix=random.choice([False, True]))
        message = self.naturalize('i have %s.' % fact_str)
        return self.message(message)

    def ask(self, facts):
        fact_str = self.fact_to_str(facts, self.num_items, include_count=False, prefix=random.choice([False, True]), question=True)
        message = self.naturalize('do you have any %s?' % fact_str)
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
            values = [v.lower() for v in item.values()]
            self.item_weights[i] += delta * len([entity for entity in entities if entity[0] in values])

    def send(self):
        # Don't send consecutive utterances with entities
        if self.sent_entity and not self.consecutive_entity:
            return None
        if self.matched_item:
            if not self.selected:
                self.selected = True
                return self.select(self.matched_item)
            else:
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
        if (not self.can_select()) or np.random.random() < 0.7:
            self.sent_entity = True
            facts, entities = self.choose_fact()
            # Decrease weights of entities mentioned
            self.update_entity_weights(entities, -10.)
            if np.random.random() < 0.5:
                return self.inform(facts)
            else:
                return self.ask(facts)
        else:
            item_id = self.sample_item()
            # Don't repeatedly select one item
            self.item_weights[item_id] = -100
            return self.select(self.kb.items[item_id])

    def can_select(self):
        '''
        We can select only when at least on item has weight > 1.
        '''
        if max(self.item_weights) > 1.:
            return True
        return False

    def is_question(self, tokens):
        first_word = tokens[0]
        last_word = tokens[-1]
        if last_word == '?' or first_word in ('do', 'does', 'what', 'any'):
            return True
        return False

    def receive(self, event):
        self.sent_entity = False
        if event.action == 'message':
            raw_utterance = event.data
            entity_tokens = self.lexicon.link_entity(tokenize(raw_utterance), kb=self.kb, mentioned_entities=self.mentioned_entities, known_kb=False)
            for token in entity_tokens:
                if is_entity(token):
                    self.mentioned_entities.add(token[1][0])
            entities = [word[1] for word in entity_tokens if is_entity(word)]

            if self.is_question(entity_tokens):
                self.asked_entities = entities

            # Update item weights
            if len(entities) > 0:
                if len([x for x in entity_tokens if x in ('no', 'none', "don't", 'zero')]) > 0:
                    negative = True
                else:
                    negative = False
                self.update_item_weights(entities, -10. if negative else 1.)

                row_entities, col_entities = self.get_related_entity(entities)
                self.update_entity_weights(entities, -10. if negative else 1.)
                self.update_entity_weights(row_entities, -1. if negative else 2.)
                self.update_entity_weights(col_entities, 1. if negative else 0.5)

        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
