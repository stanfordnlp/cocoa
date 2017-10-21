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

class Fact(object):
    """Data structure for a fact given the KB.
    """
    def __init__(self, constraint=None, entity=None, count=None, items=None):
        """
        Args:
            constraint (list[CanonicalEntity]): constraint to select a subset of items
            items: items satisfying the constraint
            entity (CanonicalEntity)
            count (int): given the constraint (i.e. among `items`), the number of items having `entity`
        """
        self.constraint = sorted(constraint)
        self.entity = entity
        self.count = count
        self.items = items

    def key(self):
        """Return a key used to identify the fact.
        """
        return tuple(self.constraint + [self.entity])


class RulebasedSession(Session):
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
        self.said_hi = False
        self.matched_item = None

        self.capitalize = random.choice([True, False])
        self.numerical = random.choice([True, False])

        self.hypothesis_set = list(kb.items)
        self.state = {
                'partner_entities': [],
                'partner_act': None,
                'my_act': None,
                'my_query': None,
                'selected': False,
                'informed_facts': set(),
                'curr_constraint': None,
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
            if item[self.attr_type_to_name[entity.type]].lower() != entity.value:
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
        values = [item[col].lower() for item in items]
        value_counts = defaultdict(int)
        for val in values:
            value_counts[val] += 1
        return value_counts

    def sort_columns(self, items, fixed_attributes):
        columns = [attr_name for attr_name in self.attr_type if not attr_name in fixed_attributes]
        print 'columns to select:', columns
        column_entropy = []
        for col in columns:
            value_counts = self.get_value_counts(items, col)
            h = self.entropy(value_counts)
            column_entropy.append((col, h))
        return sorted(column_entropy, key=lambda x: x[1])

    def select_facts_from_column(self, constraint, items, attr_name):
        value_counts = self.get_value_counts(items, attr_name)
        type_ = self.attr_type[attr_name]
        facts = []
        for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
            entity = CanonicalEntity(value, type_)
            fact = Fact(constraint=constraint, entity=entity, items=items, count=count)
            if not fact.key() in self.state['informed_facts']:
                facts.append(fact)
        return facts[:2]

    def select_facts(self, constraints, n=2):
        print 'select_facts'
        facts = []
        for constraint in constraints:
            print 'constraint:', constraint
            items = self.get_subset([constraint])
            if not items:
                facts.append(Fact(constraint=constraint, count=0))
            elif len(constraint) == len(self.kb.attributes):
                facts.append(Fact(constraint=constraint, items=items))
            else:
                fixed_attributes = [self.attr_type_to_name[entity.type] for entity in constraint]
                print 'fixed:', fixed_attributes
                for attr_name, entropy in self.sort_columns(items, fixed_attributes):
                    col_facts = self.select_facts_from_column(constraint, items, attr_name)
                    if col_facts:
                        facts.extend(col_facts)
                        break
        return facts[:n]

    def entities_to_constraints(self, entities):
        """Decide relations between multiple entities heuristically.

        If more than one entity is mentioned, we need to know whether they mean
        X AND Y and X OR Y. We follow the simple heuristic: if one entity and a
        previous entity belong to >1 items, we assume a joint relation, otherwise
        they are disjoint.

        """
        constraints = []
        for entity in entities:
            joint = False
            for constraint in constraints:
                if entity.type not in [e.type for e in constraint]:
                    constraint.append(entity)
                    joint = True
                    break
            if not joint:
                constraints.append([entity])
        return constraints

    def combine_constraints(self, constraints, prev_facts):
        new_constraints = []
        for constraint in constraints:
            print 'constraint:', constraint
            combined = False
            for fact in prev_facts:
                if fact.count > 0:
                    new_constraint = tuple(set(constraint + fact.constraint + [fact.entity]))
                    print 'new constraint:', new_constraint
                    if len(self.get_subset([new_constraint])) > 0:
                        new_constraints.append(new_constraint)
                        print 'combined'
                        combined = True
            if not combined:
                new_constraints.append(constraint)
        return new_constraints


    def fact_to_str(self, fact, include_count=True, prefix=False, question=False):
        fact_str = []
        total = self.num_items
        constraint_str = ' and '.join([entity.value for entity in fact.constraint])
        if fact.count == 0:
            return 'no {}'.format(constraint_str)
        # TODO:
        entities = [(fact.entity, fact.count)]
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
        fact_str = re.sub(r'[ ]{2,}', ' ', fact_str).strip()

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

    def guess(self, facts):
        for fact in facts:
            if len(fact.constraint) == len(self.kb.attributes):
                return random.choice(fact.items)
        return None

    def inform(self, constraints, ask=True):
        facts = self.select_facts(constraints, n=2)
        guessed_item = self.guess(facts)
        if guessed_item:
            print 'select guessed'
            return self.select(guessed_item)

        if not facts or sum([fact.count for fact in facts if fact.count is not None]) == 0:
            facts.extend(self.select_facts([[]], n=1))

        self.state['informed_facts'].update([fact.key() for fact in facts])
        self.state['my_query'] = facts
        messages = []
        if random.random() < 0.5 or (ask is False or not fact.entity == 0):
            for fact in facts:
                fact_str = self.fact_to_str(fact, prefix=random.choice([False, True]))
                message = 'i have {}.'.format(fact_str)
                messages.append(message)
        else:
            for fact in facts:
                fact_str = self.fact_to_str(fact, include_count=False, prefix=random.choice([False, True]), question=True)
                message = 'do you have any {}?'.format(fact_str)
                messages.append(message)
        return self.message(' '.join(messages))

    def answer(self, constraints):
        return self.inform(constraints, ask=False)

    def send(self):
        if self.matched_item:
            if not self.state['selected']:
                self.state['selected'] = True
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
            constraints = self.combine_constraints(constraints, self.state['my_query'])
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

            if self.is_question(entity_tokens):
                self.state['partner_act'] = 'ask'
            else:
                self.state['partner_act'] = None
            print 'received entities:', entities
            print 'partner act:', self.state['partner_act']

            if self.is_neg(entity_tokens):
                if len(entities) == 0:
                    for fact in self.state['my_query']:
                        constraint_ = fact.constraint + [fact.entity]
                        self.exclude([constraint_])
                elif len(entities) == 1:
                    constraint_ = [entity]
                    self.exclude([constraint_])
                else:
                    self.state['partner_entities'] = entities
            else:
                self.state['partner_entities'] = entities
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
