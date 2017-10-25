import random
import re
from collections import defaultdict
import numpy as np
from itertools import izip, ifilter

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
        self.constraint = constraint
        self.entity = entity
        self.count = count
        self.items = items

    def key(self):
        """Return a key used to identify the fact.
        """
        return tuple(sorted(self.to_constraint()))

    def to_constraint(self):
        if self.entity:
            return self.constraint + [self.entity]
        else:
            return self.constraint

class RulebasedSession(Session):
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

        self.hypothesis_set = list(kb.items)
        self.state = {
                'partner_entities': [],
                'partner_act': None,
                'my_act': None,
                'my_query': None,
                'selected': False,
                'informed_facts': set(),
                'curr_constraint': None,
                'selected_items': [],
                }

    def get_subset(self, constraint):
        subset = [item for item in self.hypothesis_set if self.satisfy(item, constraint)]
        return subset

    def satisfy(self, item, constraint):
        """Whether the `item` contains entities in the `constraint`.
        """
        for entity in constraint:
            if item[self.attr_type_to_name[entity.type]].lower() != entity.value:
                return False
        return True

    #def entropy(self, value_counts):
    #    h = 0
    #    total = len(value_counts)
    #    for val, count in value_counts.iteritems():
    #        p = float(count) / total
    #        h -= p * np.log(p)
    #    return h

    #def sort_columns(self, items, fixed_attributes):
    #    columns = [attr_name for attr_name in self.attr_type if not attr_name in fixed_attributes]
    #    column_entropy = []
    #    for col in columns:
    #        value_counts = self.get_value_counts(items, col)
    #        h = self.entropy(value_counts)
    #        column_entropy.append((col, h))
    #    return sorted(column_entropy, key=lambda x: x[1])

    def get_value_counts(self, items, col):
        values = [item[col].lower() for item in items]
        value_counts = defaultdict(int)
        for val in values:
            value_counts[val] += 1
        return value_counts

    def select_facts_from_column(self, constraint, items, attr_name, n=2):
        value_counts = self.get_value_counts(items, attr_name)
        type_ = self.attr_type[attr_name]
        facts = []
        for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
            entity = CanonicalEntity(value, type_)
            fact = Fact(constraint=constraint, entity=entity, items=items, count=count)
            if not fact.key() in self.state['informed_facts']:
                facts.append(fact)
        return facts[:n]

    def select_facts(self, constraints, n=2):
        print 'select_facts'
        facts = []
        optional_facts = []
        for constraint in constraints:
            print 'constraint:', constraint
            items = self.get_subset(constraint)
            if not items or len(constraint) == len(self.kb.attributes):
                facts.append(Fact(constraint=constraint, count=len(items), items=items))
            else:
                fixed_attributes = [self.attr_type_to_name[entity.type] for entity in constraint]
                print 'fixed:', fixed_attributes
                #for attr_name, entropy in self.sort_columns(items, fixed_attributes):
                for attr_name in ifilter(lambda x: not x in fixed_attributes, self.attr_type):
                    col_facts = self.select_facts_from_column(constraint, items, attr_name)
                    optional_facts.extend(col_facts)
        optional_facts.sort(key=lambda fact: fact.count, reverse=True)
        return facts + optional_facts[:1]

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
                    new_constraint = list(set(constraint + fact.to_constraint()))
                    print 'new constraint:', new_constraint
                    if len(self.get_subset(new_constraint)) > 0:
                        new_constraints.append(new_constraint)
                        print 'combined'
                        combined = True
            if not combined:
                new_constraints.append(constraint)
        return new_constraints

    def realize(self, entity):
        return self.realizer._realize_entity(entity).surface

    def fact_to_str(self, fact, include_count=True, prefix=False, question=False):
        fact_str = []
        total = self.num_items
        constraint_str = ' and '.join([entity.value for entity in fact.constraint])
        if fact.count == 0:
            return 'no {}'.format(constraint_str)
        # TODO:
        if len(fact.constraint) > 1:
            prefix = True
        entities = [(fact.entity, fact.count)]
        for i, (entity, count) in enumerate(entities):
            entity_str = self.realize(entity)
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

    def guess(self, facts):
        for fact in facts:
            if len(fact.items) > 0 and len(fact.constraint) == len(self.kb.attributes):
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
        def _inform(i, fact):
            if fact.count == 0 and len(fact.constraint) > 2:
                return 'nope.'
            fact_str = self.fact_to_str(fact, prefix=random.choice([False, True]))
            prefix = 'i have' if i == 0 and fact.count != 0 else ''
            message = '{prefix} {fact}.'.format(prefix=prefix, fact=fact_str)
            return message
        def _ask(i, fact):
            fact_str = self.fact_to_str(fact, include_count=False, prefix=random.choice([False, True]), question=True)
            prefix = 'any' if i == 0 else 'or'
            message = '{prefix} {fact}?'.format(prefix=prefix, fact=fact_str)
            return message
        # TODO:
        facts = sorted(facts, key=lambda x: x.count)
        if len(facts) == 1 and len(facts[0].constraint) > 1:
            messages.append('{}?'.format(self.realize(facts[0].entity)))
        else:
            for i, fact in enumerate(facts):
                if ask is False or fact.count == 0:
                    messages.append(_inform(i, fact))
                elif i > 0 and facts[0].count == 0:
                    messages.append(_ask(0, fact))
                else:
                    messages.append(_inform(i, fact))
        return self.message(' '.join(messages))

    def answer(self, constraints):
        return self.inform(constraints, ask=False)

    def send(self):
        if self.matched_item:
            if self.matched_item in self.state['selected_items']:
                return self.wait()
            else:
                return self.select(self.matched_item)

        if len(self.state['partner_entities']) > 0:
            constraints = self.entities_to_constraints(self.state['partner_entities'])
            partner_facts = [Fact(constraint=c) for c in constraints]
            self.state['informed_facts'].update([f.key() for f in partner_facts])
            if self.state['partner_act'] in ('ask', 'inform'):
                constraints = self.combine_constraints(constraints, self.state['my_query'])
        else:
            constraints = [[]]

        # Reply to questions
        if self.state['partner_act'] == 'ask' and self.state['partner_entities']:
            return self.answer(constraints)

        if len(self.hypothesis_set) > 2:
            return self.inform(constraints)
        else:
            items = [item for item in self.hypothesis_set if not item in self.state['selected_items']]
            # Something went wrong...Reset.
            if len(items) == 0:
                self.hypothesis_set = self.kb.items
                return self.inform(constraints)
            item = random.choice(items)
            self.state['selected_items'].append(item)
            return self.select(item)

        raise Exception('Uncatched case')

    def is_question(self, tokens):
        first_word = tokens[0]
        last_word = tokens[-1]
        if last_word == '?' or first_word in ('do', 'does', 'what', 'any'):
            return True
        return False

    def exclude(self, constraint):
        print 'EXCLUDE:', constraint
        items = []
        for item in self.hypothesis_set:
            if not self.satisfy(item, constraint):
                items.append(item)
        if items:
            self.hypothesis_set = items
        else:
            # Something went wrong... Reset.
            self.hypothesis_set = kb.items

    # TODO: add to prev entities
    #def select(self, item):

    def parse_entities(self, entity_tokens):
        neg = False
        has_neg = False
        prev_entities = set([e for fact in self.state['my_query'] for e in fact.to_constraint()])
        exclude_entities = []
        entities = []
        neg_words = ('no', "don't", 'not', 'none', 'zero', 'nope', 'nobody')
        sentence_delimiter = ('.', ',', ';')
        for i, token in enumerate(entity_tokens):
            if token in neg_words:
                neg = True
                has_neg = True
            elif token in sentence_delimiter:
                neg = False
            elif is_entity(token):
                entity = token.canonical
                if neg and (entity in prev_entities or (i > 0 and entity_tokens[i-1] in neg_words)):
                    exclude_entities.append(entity)
                elif entity not in entities:
                    entities.append(entity)
        if has_neg and not exclude_entities:
            exclude_constraints = [fact.to_constraint() for fact in self.state['my_query']]
        else:
            exclude_constraints = self.entities_to_constraints(exclude_entities)
        # Agree
        if not has_neg and not entities and not self.is_question(entity_tokens):
            entities = prev_entities
        return entities, exclude_constraints

    def receive(self, event):
        if event.action == 'message':
            raw_utterance = event.data
            entity_tokens = self.lexicon.link_entity(tokenize(raw_utterance), kb=self.kb, mentioned_entities=self.mentioned_entities, known_kb=False)

            entities, exclude_constraints = self.parse_entities(entity_tokens)
            self.mentioned_entities.update([entity.value for entity in entities])
            for constraint in exclude_constraints:
                self.exclude(constraint)

            self.state['partner_entities'] = entities
            if self.is_question(entity_tokens):
                self.state['partner_act'] = 'ask'
            else:
                self.state['partner_act'] = 'inform'

            print 'RECEIVE:', self.state['partner_act']
            print entity_tokens
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
