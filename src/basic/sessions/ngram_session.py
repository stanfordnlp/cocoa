import json
import datetime

__author__ = 'anushabala'

from src.basic.sessions.session import Session
from src.model.preprocess import markers
from src.basic.ngram_util import preprocess_event
from src.basic.event import Event
from src.basic.tagger import SelectionFeatures


class NgramSession(Session):
    _MAX_TOKENS = 50
    """
    This class represents a dialogue agent that uses an underlying ngram model to generate utterances. The
    NgramSystem class is the class that actually "trains" the ngram model from a set of training examples,
    while NeuralSession represents an instance of the model as used in the context of a dialogue.
    """
    def __init__(self, agent, scenario, uuid, type_attribute_mappings, lexicon, tagger, executor, model):
        super(NgramSession, self).__init__(agent)
        self.scenario = scenario
        self.uuid = uuid

        self.type_attribute_mappings = type_attribute_mappings
        self.lexicon = lexicon
        self.tagger = tagger
        self.model = model
        self.executor = executor

        self.tagged_history = []
        self.history = []

    def send(self):
        if len(self.history) == 0:
            self.update_history(self.agent, [markers.EOS])
        generated = self.generate()
        # print "Generated tagged tokens:", generated
        event = self.convert_generated_tokens_to_event(generated)
        # print "Generated event=%s" % event.action
        return event

    def convert_generated_tokens_to_event(self, generated_tokens):
        if generated_tokens[0] == markers.SELECT:
            item = json.loads(generated_tokens[1][0])
            return Event.SelectionEvent(self.agent, item, datetime.datetime.now())

        raw_tokens = []
        for token in generated_tokens[:-1]:
            if isinstance(token, tuple):
                raw, _ = token
                if raw is None:
                    raw_tokens.append("")
                else:
                    raw_tokens.append(raw)
            else:
                raw_tokens.append(token)

        return Event.MessageEvent(self.agent, " ".join(raw_tokens), datetime.datetime.now())

    def generate(self):
        token = None
        generated_tokens = []
        while token != markers.EOS and len(generated_tokens) <= self._MAX_TOKENS:
            token = self.model.generate(self.history, preprocess=True)
            token_with_entity = self.add_entities(token)
            generated_tokens.append(token_with_entity)
            self.history.append(token_with_entity)

        self.tagged_history.append((self.agent, generated_tokens))
        return generated_tokens

    def add_entities(self, token):
        if isinstance(token, tuple):
            entity_type, features = token
            raw, canonical = self.executor.get_entity(token, self.agent, self.scenario, self.tagged_history)
            return raw, (canonical, entity_type, features)
        return token

    def receive(self, event):
        agent, tokens = preprocess_event(event)
        # print "Received (event=%s)" % event.action
        # print "Received tokens: ", tokens
        if event.action == 'select':
            tagged_tokens = self.tagger.tag_selection(self.agent, self.scenario, tokens)
        else:
            linked_tokens = self.lexicon.link_entity(tokens, agent=self.agent, uuid=self.uuid)
            tagged_tokens = self.tagger.tag_utterance(linked_tokens, self.scenario, self.agent, self.tagged_history)
        # print "Tagged received tokens:", tagged_tokens
        self.update_history(event.agent, tagged_tokens)

    def update_history(self, agent_idx, tagged_tokens):
        self.tagged_history.append((agent_idx, tagged_tokens))
        self.history.extend(tagged_tokens)
