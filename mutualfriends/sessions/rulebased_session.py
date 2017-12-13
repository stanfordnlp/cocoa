import random
import re
from collections import defaultdict
import numpy as np
from itertools import izip

from cocoa.model.parser import LogicalForm as LF
from cocoa.core.entity import Entity, CanonicalEntity
from cocoa.core.sample_utils import sample_candidates
from cocoa.sessions.rulebased_session import RulebasedSession as BaseRulebasedSession

from core.tokenizer import tokenize
from model.parser import Parser, Utterance
from model.dialogue_state import DialogueState

class RulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager, realizer=None):
        parser = Parser(agent, kb, lexicon)
        state = DialogueState(agent, kb)
        super(RulebasedSession, self).__init__(agent, kb, parser, generator, manager, state, sample_temperature=5.)

        self.kb = kb
        self.attr_type = {attr.name: attr.value_type for attr in kb.attributes}
        self.num_items = len(kb.items)
        self.entity_counts = self.count_entity()
        self.entity_coords = self.get_entity_coords()
        self.entity_weights = self.init_entity_weight()
        self.item_weights = [1.] * self.num_items

        self.realizer = realizer

    def get_entity_coords(self):
        '''
        Return a dict of {entity: [row]}
        '''
        entity_coords = defaultdict(list)
        for row, item in enumerate(self.kb.items):
            for col, attr in enumerate(self.kb.attributes):
                entity = CanonicalEntity(value=item[attr.name].lower(), type=attr.value_type)
                entity_coords[entity].append(row)
        return entity_coords

    def get_row_entities(self, entities):
        row_entities = set()
        for entity in entities:
            rows = self.entity_coords[entity]
            for row, item in enumerate(self.kb.items):
                for col, attr in enumerate(self.kb.attributes):
                    if row in rows:
                        e = CanonicalEntity(item[attr.name].lower(), attr.value_type)
                        if e not in entities:
                            row_entities.add(e)
        return row_entities

    def count_entity(self):
        '''
        Return a dict of {entity: count}.
        '''
        entity_counts = defaultdict(int)
        for item in self.kb.items:
            for attr_name, entity_value in item.iteritems():
                entity = CanonicalEntity(entity_value.lower(), self.attr_type[attr_name])
                entity_counts[entity] += 1
        return entity_counts

    def init_entity_weight(self):
        entity_counts = self.entity_counts
        N = float(self.num_items)
        # Scale counts to [0, 1]
        entity_weights = {entity: count / N for entity, count in entity_counts.iteritems()}
        return entity_weights

    def sample_item(self):
        for i, item in enumerate(self.kb.items):
            if item in self.state.selected_items:
                self.item_weights[i] = -100.
        item_id = np.argmax(self.item_weights)
        return item_id

    def update_entity_weights(self, entities, delta):
        for entity in entities:
            if entity in self.entity_weights:
                self.entity_weights[entity] += delta

    def update_item_weights(self, entities, delta):
        for i, item in enumerate(self.kb.items):
            values = [v.lower() for v in item.values()]
            self.item_weights[i] += delta * len([entity for entity in entities if entity.value.lower() in values])

    def _get_entity_types(self, template):
        types = []
        for token in template.split():
            if token[0] == '{' and token[-1] == '}':
                types.append(token[1:-1])
        return types

    def fill_entities(self, types):
        ents = []
        for type_ in types:
            entities = [(e, w - 10. if e in self.state.recent_mentioned_entities else 0.) for e, w in self.entity_weights.iteritems() if e.type == type_]
            entity = sorted(entities, key=lambda x: x[1], reverse=True)[0][0]
            ents.append(entity)
        return ents

    def template_message(self, intent):
        template = self.retrieve_response_template(intent)
        #entity_types = self._get_entity_types(template['template'])
        #entities = self.fill_entities(entity_types)
        #lf = LF(intent, entities=entities)
        #text = self.fill_template(template, entities)
        lf = LF(intent)
        text = template['template']
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

    def realize(self, entity):
        if self.realizer:
            s = self.realizer.realize_entity(entity)
        else:
            s = entity.value
        return s

    def fill_template(self, template, entities):
        template = template['template']
        type_to_entities = defaultdict(list)
        for ent in entities:
            type_to_entities[ent.type].append(self.realize(ent))
        kwargs = dict(type_to_entities)
        if len(entities) == 1:
            count = self.entity_counts[entities[0]]
            kwargs['number'] = self.number_to_str(count, len(self.kb.items))
        print kwargs
        print template
        text = template.format(**kwargs)
        return text

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

    def inform(self, intent):
        entities = self.choose_entities()
        print 'choose entities:', entities
        if len(entities) == 1 and self.entity_counts[entities[0]] == 0:
            intent = 'negative'
        signature = self.parser.signature(entities)
        template = self.retrieve_response_template(intent, signature=signature)
        text = self.fill_template(template, entities)
        lf = LF(intent, entities=entities)
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

    #def get_min_subset(self, entities):
    #    subset = None
    #    prefix = []
    #    for e in entities:
    #        if not subset:
    #            subset = self.get_row_entities([e])
    #        else:
    #            new_subset = subset.intersection(self.get_row_entities([e]))
    #            if len(new_subset) == 0:
    #                return subset, prefix
    #        prefix.append(e)
    #    return subset, prefix

    def choose_entities(self):
        if self.state.partner_entities:
            #subset, prefix = self.get_min_subset(self.state.partner_entities)
            #print 'prefix:', prefix
            #print 'subset:', subset
            #if subset:
            partner_entity = random.choice(self.state.partner_entities)
            row_entities = self.get_row_entities([partner_entity])
            if not row_entities:
                return [partner_entity]
            entities = [partner_entity] + [random.choice(list(row_entities))]
            #entities = random.choice(prefix) + random.choice(list(subset))
            #print 'correct:', entities
            return entities

        entities = [(e, w - (10. if e in self.state.recent_mentioned_entities else 0.))
                for e, w in self.entity_weights.iteritems()]
        entities = sorted(entities, key=lambda x: x[1], reverse=True)
        entities = [x[0] for x in entities]
        entities = entities[:1]
        return entities

    def select(self):
        if self.state.matched_item:
            if not self.has_done('select'):
                return super(RulebasedSession, self).select(self.state.matched_item)
            else:
                return self.wait()
        else:
            item_id = self.sample_item()
            return super(RulebasedSession, self).select(self.kb.items[item_id])

    def send(self):
        action = self.manager.choose_action(state=self.state)
        if not action:
            action = self.retrieve_action()
        if action in ('inform', 'inquire'):
            return self.inform(action)
        elif action == 'select':
            return self.select()
        else:
            return self.template_message(action)

        raise Exception('Uncaught case')

    def receive(self, event):
        super(RulebasedSession, self).receive(event)
        if self.state.partner_exclude_entities:
            self.update_item_weights(self.state.partner_exclude_entities, -1.)
            self.update_entity_weights(self.state.partner_exclude_entities, -10.)
        if self.state.partner_entities:
            row_entities = self.get_row_entities(self.state.partner_entities)
            print 'update weights:', [str(x) for x in row_entities]
            self.update_entity_weights(row_entities, 1.)
