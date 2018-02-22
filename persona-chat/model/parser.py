import re
import copy

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, LogicalForm as LF, Utterance

from core.tokenizer import tokenize

class Parser(BaseParser):
    ME = 0
    YOU = 1

    def is_bye(self, utterance):
        s = utterance.text
        if 'bye' in utterance.tokens or re.search(r'(great|nice|fun) (talking|chatting)', s):
            return True
        return False

    def get_entities(self, tokens, type_=None):
        entities = [x for x in tokens if is_entity(x) and (type_ is None or x.canonical.type == type_)]
        return entities

    def extract_template(self, tokens, dialogue_state):
        return [x.surface if is_entity(x) else x for x in tokens]

    def parse(self, event, dialogue_state):
        # We are parsing the partner's utterance
        assert event.agent == 1 - self.agent
        if event.action == 'done':
            u = self.parse_action(event)
        elif event.action == 'message':
            u = self.parse_message(event, dialogue_state)
        else:
            return False

        return u

    def parse_message(self, event, dialogue_state):
        tokens = self.lexicon.link_entity(event.data)
        tokens = [x.lower() if not is_entity(x) else x for x in tokens]
        utterance = Utterance(raw_text=event.data, tokens=tokens)
        intent = "placeholder_intent"
        template = self.extract_template(tokens, dialogue_state)
        utterance.lf = LF(intent, topic="placeholder")
        utterance.template = template
        return utterance