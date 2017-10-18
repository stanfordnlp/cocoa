import random
import re
from collections import defaultdict
import numpy as np
from itertools import izip

from cocoa.core.sample_utils import sample_candidates
from cocoa.core.entity import is_entity, CanonicalEntity

from model.preprocess import tokenize, word_to_num
from core.lexicon import Lexicon
from session import Session

num_to_word = {v: k for k, v in word_to_num.iteritems()}

class RulebasedSession(Session):
    '''
    The simple system implements a bot that
    - greets
    - asks or informs about a fact in its KB
    - replies 'no' or 'yes' to partner's repsonse
    - selects and item
    '''

    greetings = ['hi', 'hello', 'hey', 'hiya']

    def __init__(self, agent, kb, lexicon, realizer):
        super(RulebasedSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.attr_type = {attr.name: attr.value_type for attr in kb.attributes}
        self.attr_type_to_name = {attr.value_type: attr.name for attr in kb.attributes}
        self.lexicon = lexicon
        self.realizer = realizer
        self.num_items = len(kb.items)

        self.mentioned_entities = set()

        # Dialogue state
        self.asked_entities = None
        self.answered = False
        self.said_hi = False
        self.matched_item = None
        self.selected = False

        self.capitalize = random.choice([True, False])
        self.numerical = random.choice([True, False])

        self.hypothesis_set = list(kb.items)
        self.state = {
                'partner_entities': [],
                'partner_act': None,
                'my_query': None,
                }

    def get_subset(self, constraints):
        subset = [item for item in self.hypothesis_set if self.satisfy(item, constraints)]
        return subset

    def satisfy(self, item, constraints):
        """Satisfy OR constraints.
        """
        if not constraints:
            return True
        for constraint in constraints:
            if self._satisfy(item, constraint):
                return True
        return False

    def _satisfy(self, item, constraint):
        """Satisfy AND constraints.
        """
        for entity in constraint:
            if item[self.attr_type_to_name[entity.type]] != entity.value:
                return False
        return True

    def entropy(self, value_counts):
        h = 0
        total = len(value_counts)
        for val, count in value_counts.iteritems():
            p = float(count) / total
            h -= p * np.log(p)
        return h

    def get_value_counts(self, items, col):
        values = [item[col] for item in items]
        value_counts = defaultdict(int)
        for val in values:
            value_counts[val] += 1
        return value_counts

    def get_lowest_entropy_column(self, items, fixed_attributes):
        columns = [attr_name for attr_name in self.attr_type if not attr_name in fixed_attributes]
        print 'columns to select:', columns
        column_entropy = []
        for col in columns:
            value_counts = self.get_value_counts(items, col)
            h = self.entropy(value_counts)
            column_entropy.append((col, h))
        return min(column_entropy, key=lambda x: x[1])

    def select_query_entities(self, constraints, max_entities=2):
        selected_columns = []
        print 'select_query_entities'
        for constraint in constraints:
            print 'constraint:', constraint
            items = self.get_subset([constraint])
            fixed_attributes = [self.attr_type_to_name[entity.type] for entity in constraint]
            print 'fixed:', fixed_attributes
            attr_name, entropy = self.get_lowest_entropy_column(items, fixed_attributes)
            selected_columns.append((attr_name, entropy, constraint, items))
        attr_name, entropy, constraint, items = min(selected_columns, key=lambda x: x[1])

        value_counts = self.get_value_counts(items, attr_name)
        type_ = self.attr_type[attr_name]
        if len(value_counts) < 3:
            entities = [(CanonicalEntity(value, type_), count) for value, count in value_counts.iteritems()]
        else:
            value, count = max(value_counts.items(), key=lambda x: x[1])
            entities = [(CanonicalEntity(value, type_), count)]
        return constraint, entities

    def entities_to_constraints(self, entities):
        """Decide relations between multiple entities heuristically.

        If more than one entity is mentioned, we need to know whether they mean
        X AND Y and X OR Y. We follow the simple heuristic: if one entity and a
        previous entity belong to >1 items, we assume a joint relation, otherwise
        they are disjoint.

        """
        constraints = []
        while len(entities) > 0:
            i = 0
            entity = entities[i]
            rowi = self.get_subset([[entity]])
            constraint = [entity]
            for j in xrange(i+1, len(entities)):
                entity2 = entities[j]
                rowj = self.get_subset([[entity2]])
                intersect = [r for r in rowi if r in rowj]
                if len(intersect) > 0:
                    constraint.append(entity2)
                    rowi = intersect
            # Remove converted entities
            entities = [entity for entity in entities if entity not in constraint]
            constraints.append(constraint)
        return constraints

    def fact_to_str(self, constraint, entities, include_count=True, prefix=False, question=False):
        fact_str = []
        total = self.num_items
        constraint_str = ' and '.join([entity.value for entity in constraint])
        if not entities:
            return 'no {}'.format(constraint_str)
        for i, (entity, count) in enumerate(entities):
            entity_str = self.realizer._realize_entity(entity).surface
            if prefix:
                if entity.type == 'name':
                    p = 'named'
                elif entity.type == 'school':
                    p = 'who went to'
                elif entity.type == 'company':
                    p = 'working at'
                elif entity.type == 'major':
                    p = 'who studied'
                else:
                    p = random.choice(['who like' if count > 1 else 'who like', 'into'])
                entity_str = '{prefix} {entity}'.format(prefix=p, entity=entity_str)

            friend = '%s' % ('friends' if count > 1 else 'friend') if prefix else ''
            count = self.number_to_str(count, total) if include_count else ''
            s = '{count} {constraint} {friend} {entity}'.format(constraint=constraint_str, count=count, friend=friend, entity=entity_str)
            fact_str.append(s)

        fact_str = ', '.join(fact_str)
        fact_str = re.sub(r'[ ]{2,}', ' ', fact_str)

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

    def inform(self, constraints, ask=True):
        constraint, entities = self.select_query_entities(constraints)
        print 'inform'
        print 'constraint:', constraint
        print 'entities:', entities
        self.state['my_query'] = (constraint, entities)
        if random.random() < 0.5 or (ask is False or len(entities) == 0):
            fact_str = self.fact_to_str(constraint, entities, prefix=random.choice([False, True]))
            message = 'i have {}.'.format(fact_str)
        else:
            fact_str = self.fact_to_str(constraint, entities, include_count=False, prefix=random.choice([False, True]), question=True)
            message = 'do you have any {}?'.format(fact_str)
        return self.message(message)

    def answer(self, constraints):
        return self.inform(constraints, ask=False)

    def send(self):
        if self.matched_item:
            if not self.selected:
                # TODO: put this in state
                self.selected = True
                return self.select(self.matched_item)
            else:
                return None

        # Say hi first
        #if not self.said_hi:
        #    self.said_hi = True
        #    text = random.choice(self.greetings)
        #    return self.message(text)

        if self.state['partner_entities']:
            constraints = self.entities_to_constraints(self.state['partner_entities'])
        else:
            constraints = [[]]

        # Reply to questions
        if self.state['partner_act'] == 'ask' and self.state['partner_entities']:
            return self.answer(constraints)

        # Inform or Ask or Select
        if len(self.hypothesis_set) > 3:
            return self.inform(constraints)
        else:
            # TODO: Don't repeatedly select one item
            item = random.choice(self.hypothesis_set)
            return self.select(item)

    def is_question(self, tokens):
        first_word = tokens[0]
        last_word = tokens[-1]
        if last_word == '?' or first_word in ('do', 'does', 'what', 'any'):
            return True
        return False

    def is_neg(self, tokens):
        if len([x for x in tokens if x in ('no', 'none', "don't", 'zero')]) > 0:
            return True
        return False

    def exclude(self, constraints):
        print 'exclude:', constraints
        items = []
        for item in self.hypothesis_set:
            if not self.satisfy(item, constraints):
                items.append(item)
        if items:
            self.hypothesis_set = items
        else:
            # Something went wrong... Reset.
            self.hypothesis_set = kb.items

    def receive(self, event):
        if event.action == 'message':
            raw_utterance = event.data
            entity_tokens = self.lexicon.link_entity(tokenize(raw_utterance), kb=self.kb, mentioned_entities=self.mentioned_entities, known_kb=False)
            print entity_tokens

            entities = [word.canonical for word in entity_tokens if is_entity(word)]
            self.mentioned_entities.update([entity.value for entity in entities])
            self.state['partner_entities'] = entities

            if self.is_question(entity_tokens):
                self.state['partner_act'] = 'ask'
            else:
                self.state['partner_act'] = None
            print 'received entities:', entities
            print 'partner act:', self.state['partner_act']

            if self.is_neg(entity_tokens):
                if len(entities) == 0:
                    constraint, entities = self.state['my_query']
                    for entity in entities:
                        constraint_ = constraint + [entity]
                        self.exclude([constraint_])
                elif len(entities) == 1:
                    constraint_ = [entity]
                    self.exclude([constraint_])
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
