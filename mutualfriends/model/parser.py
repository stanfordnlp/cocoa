from collections import defaultdict

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, LogicalForm as LF, Utterance

from core.tokenizer import tokenize

class Parser(BaseParser):
    neg_words = ('no', "don't", 'not', 'none', 'zero', 'nope', 'nobody')
    numbers = ('one', '1', 'two', '2', 'three', '3', 'four', '4', 'five', '5', 'six', '6', 'seven', '7', 'eight', '8', 'nine', '9', 'ten', '10')

    def parse_select(self, event):
        matched = False
        for item in self.kb.items:
            if item == event.data:
                matched = True
        return Utterance(logical_form=LF(event.action, item=event.data, matched=matched), template=['<select>'])

    def has_entity(self, utterance):
        for token in utterance.tokens:
            if is_entity(token):
                return True
        return False

    def classify_intent(self, utterance):
        if self.is_question(utterance):
            return 'inquire'
        elif self.has_entity(utterance):
            return 'inform'
        elif self.is_negative(utterance):
            return 'negative'
        elif self.is_greeting(utterance):
            return 'greet'
        else:
            return 'unknown'

    def extract_template(self, tokens, dialogue_state):
        template = []
        type_count = defaultdict(int)
        for token in tokens:
            if token in self.numbers or token in ('no', 'all'):
                template.append('{number}')
            elif is_entity(token):
                type_ = token.canonical.type
                template.append('{{{0}[{1}]}}'.format(type_, type_count[type_]))
                type_count[type_] += 1
            else:
                template.append(token)
        return template

    def default_template(self, entities):
        type_count = defaultdict(int)
        template = []
        for entity in entities:
            type_ = entity.type
            template.append('{{{0}[{1}]}}'.format(type_, type_count[type_]))
            type_count[type_] += 1
        return {'tempalte': ', '.join(template)}

    def signature(self, entities):
        return '|'.join(sorted([e.type for e in entities]))

    def parse_message(self, event, dialogue_state):
        tokens = self.lexicon.link_entity(tokenize(event.data), kb=self.kb, mentioned_entities=dialogue_state.mentioned_entities, known_kb=False)
        utterance = Utterance(raw_text=event.data, tokens=tokens)
        intent = self.classify_intent(utterance)

        exclude_entities = []
        entities = []
        for i, token in enumerate(tokens):
            if is_entity(token):
                if i > 0 and tokens[i-1] in self.neg_words:
                    exclude_entities.append(token.canonical)
                else:
                    entities.append(token.canonical)

        if len(entities) == 0 and len(exclude_entities) > 0:
            intent = 'negative'

        signature = ''
        if self.is_negative(utterance) and intent == 'inform':
            utterance.ambiguous_template = True
        elif entities:
            signature = self.signature(entities)
        elif exclude_entities:
            signature = self.signature(exclude_entities)

        if intent == 'negative' and not exclude_entities:
            exclude_entities = dialogue_state.my_entities

        lf = LF(intent, entities=entities, exclude_entities=exclude_entities, signature=signature)
        utterance.lf = lf

        utterance.template = self.extract_template(tokens, dialogue_state)

        return utterance

    def parse(self, event, dialogue_state):
        # We are parsing the partner's utterance
        assert event.agent == 1 - self.agent
        if event.action == 'select':
            u = self.parse_select(event)
        elif event.action == 'message':
            u = self.parse_message(event, dialogue_state)
        else:
            return False

        return u

