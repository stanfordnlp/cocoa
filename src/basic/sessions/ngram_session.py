from collections import defaultdict
from datetime import datetime
from src.basic.ngram_model import NgramModel
from src.basic.tagger import Tagger

__author__ = 'anushabala'

from src.basic.sessions.session import Session
from src.model.preprocess import markers, is_entity
from src.basic.ngram_util import preprocess_event
from src.basic.event import Event
import itertools as it


class NgramSession(Session):
    _MAX_TOKENS = 50
    DEFAULT_MODEL = 'default'
    """
    This class represents a dialogue agent that uses an underlying ngram model to generate utterances. The
    NgramSystem class is the class that actually "trains" the ngram model from a set of training examples,
    while NeuralSession represents an instance of the model as used in the context of a dialogue.
    """
    def __init__(self, agent, scenario, uuid, type_attribute_mappings, lexicon, tagger, executor, models):
        super(NgramSession, self).__init__(agent)
        self.scenario = scenario
        self.kb = scenario.get_kb(agent)
        self.uuid = uuid

        self.type_attribute_mappings = type_attribute_mappings
        self.lexicon = lexicon
        self.tagger = tagger

        attributes = tuple(sorted([attr.name for attr in scenario.attributes]))
        self.model = self._get_model(models, attributes)

        self.executor = executor

        self.tagged_history = []
        self.history = []

        self.entity_scores = defaultdict(lambda : defaultdict(int))
        for item in self.kb.items:
            for attr_name, attr_value in item.iteritems():
                self.entity_scores[attr_name][attr_value] = 1.

    def _get_model(self, models, attributes):
        if attributes in models.keys():
            return models[attributes]
        return models[self.DEFAULT_MODEL]

    def send(self):
        if len(self.history) == 0:
            self.update_history(1 - self.agent, [markers.GO, markers.EOS])
        generated = self.generate()
        # print "Generated tagged tokens:", generated
        event = self.convert_generated_tokens_to_event(generated)
        # print "Generated event=%s" % event.action
        return event

    def convert_generated_tokens_to_event(self, generated_tokens):
        timestamp = (datetime.now() - datetime.fromtimestamp(0)).total_seconds()
        if generated_tokens[0] == markers.SELECT:
            # print generated_tokens
            item = generated_tokens[1][0]
            # print item
            return Event.SelectionEvent(self.agent, item, timestamp, metadata=generated_tokens)

        raw_tokens = []
        for token in generated_tokens[:-1]:
            if is_entity(token):
                raw, _ = token
                if raw is None:
                    raw_tokens.append("")
                else:
                    raw_tokens.append(raw)
            else:
                raw_tokens.append(token)
        return Event.MessageEvent(self.agent, " ".join(raw_tokens), timestamp, metadata=generated_tokens)

    def is_selection(self, token):
        # print token
        if is_entity(token):
            entity_type, _ = token
            return entity_type == Tagger.SELECTION_TYPE
        return False

    def is_selection_valid(self, generated_tokens):
        # If selection, generated tokens must be list of length 3 of form: [selection marker, selected item, EOS marker]
        # selected item must be a dictionary
        if len(generated_tokens) != 3:
            return False

        if generated_tokens[0] != markers.SELECT:
            return False

        if not self.is_selection(generated_tokens[1]):
            return False

        if generated_tokens[2] != markers.EOS:
            return False

        return True

    def is_generation_valid(self, generated_tokens):
        for (i, token) in enumerate(generated_tokens):
            if not self.is_token_valid(token):
                return False
            if token == markers.SELECT:
                return self.is_selection_valid(generated_tokens)
            if self.is_selection(token):
                return False

            if is_entity(token):
                candidates = self.get_candidates(token)
                if candidates is None:
                    return False

        return True

    def generate(self, retry_limit=10):
        def _add_entities_to_utterance(tokens):
            tagged_tokens = []
            for t in tokens:
                token_with_entity = t
                if is_entity(t):
                    candidates = self.get_candidates(t)
                    token_with_entity = self.add_entity(t, candidates)
                tagged_tokens.append(token_with_entity)
            return tagged_tokens

        retries = 0
        generation_valid = False
        generated_tokens = []
        while retries < retry_limit and not generation_valid:
            token = None
            while token != markers.EOS and len(generated_tokens) <= self._MAX_TOKENS:
                token = self.model.generate(self.history, preprocess=False)
                generated_tokens.append(token)
                self.history.append(token)

            if self.is_generation_valid(generated_tokens):
                generation_valid = True
            else:
                retries += 1
                if retries < retry_limit:
                    generated_tokens = []
                    self.undo_generated_utterance()

        tokens_with_entities = _add_entities_to_utterance(generated_tokens)
        self.tagged_history.append((self.agent, tokens_with_entities))
        # print tokens_with_entities
        return tokens_with_entities

    def undo_generated_utterance(self):
        """
        Resets the history of the model to remove the last generated utterance
        :return:
        """
        indices = xrange(len(self.history)-1, -1, -1)
        gen = it.izip(indices, reversed(self.history))
        try:
            pos = next(i for i, value in gen if value == markers.EOS)
        except StopIteration:
            pos = -1
        self.history = self.history[:pos+1]

    def is_token_valid(self, token):
        if is_entity(token):
            entity_type, features = token
            return entity_type in [attr.value_type for attr in self.kb.attributes] or entity_type == Tagger.SELECTION_TYPE
        return True

    def get_candidates(self, token):
        candidates = self.executor.get_candidate_entities(token, self.agent, self.scenario, self.tagged_history)
        if candidates is None:
            return None
        return candidates

    def add_entity(self, token, candidates):
        entity_type, features = token
        if candidates is None:
            # candidate set is None when entity type isn't valid or when features lead to impossible state (e.g.
            # mention feature present but no entities were mentioned
            if not self.is_token_valid(token):
                # if entity type isn't valid return none
                # print "[Agent %d]  Invalid token: " % self.agent, token
                return ("NONE", ("NONE", entity_type, features))
            else:
                # if features lead to invalid state just choose best ranked entity of this type
                # print "[Agent %d]  Invalid state from features:" % self.agent, token
                if entity_type == Tagger.SELECTION_TYPE:
                    candidates = self.kb.items
                else:
                    attr_name = self.type_attribute_mappings[entity_type]
                    candidates = [x[attr_name] for x in self.kb.items]

        best_candidate, score = self.select_candidate_from_type(entity_type, candidates)
        self.update_mentioned(entity_type, best_candidate)
        return best_candidate, (best_candidate, entity_type, features)

    def select_candidate_from_type(self, entity_type, candidates):
        if entity_type == Tagger.SELECTION_TYPE:
            return self.select_best_candidate(candidates)
        attr_name = self.type_attribute_mappings[entity_type]
        candidate_scores = [(c, self.entity_scores[attr_name][c]) for c in candidates]
        # print "[Agent %d] Sorted candidates (type %s):" % (self.agent, entity_type)
        # print list(sorted(candidate_scores, key=lambda x: x[1], reverse=True))
        return max(candidate_scores, key=lambda x: x[1])

    def select_best_candidate(self, candidates):
        # select best candidate based on score across attributes
        total_scores = {i: 0 for i in range(0, len(candidates))}
        for attr in self.kb.attributes:
            for idx, c in enumerate(candidates):
                attr_value = c[attr.name]
                total_scores[idx] += self.entity_scores[attr.name][attr_value]
        # return candidate with highest score
        # print "[Agent %d] Item candidates: " % self.agent, candidates
        # print "[Agent %d] Sorted order:" % self.agent
        # print list(sorted(total_scores.items(), key=lambda x: x[1], reverse=True))
        best_idx, score = max(total_scores.items(), key=lambda x: x[1])
        return candidates[best_idx], score

    def update_scores(self, tagged_tokens):
        for token in tagged_tokens:
            if is_entity(token):
                (_, (entity, entity_type, _)) = token
                if entity_type == Tagger.SELECTION_TYPE:
                    # print "[Agent %d] Updating score for all attributes of selected entity" % self.agent, entity
                    # print type(entity)
                    for attr_name, attr_value in entity.items():
                        self.entity_scores[attr_name][attr_value] += 1.
                else:
                    attr_name = self.type_attribute_mappings[entity_type]
                    # print "[Agent %d] Updating score for mentioned attribute %s (type %s)" % (self.agent, entity, attr_name)
                    self.entity_scores[attr_name][entity] += 1.

    def update_mentioned(self, entity_type, mentioned_entity):
        if entity_type != Tagger.SELECTION_TYPE:
            attr_name = self.type_attribute_mappings[entity_type]
            # print "[Agent %d] Mentioned attribute %s (type %s); setting score to 0" % (self.agent, mentioned_entity, attr_name)
            self.entity_scores[attr_name][mentioned_entity] = 0.

    def receive(self, event):

        if event is not None and event.data is not None:
            agent, tokens = preprocess_event(event)
            # print "Received (event=%s)" % event.action
            # print "Received tokens: ", tokens
            if event.action == 'select':
                tagged_tokens = self.tagger.tag_selection(self.agent, self.scenario, tokens)
            else:
                linked_tokens = self.lexicon.link_entity(tokens, uuid=self.uuid, kb=self.scenario.get_kb(1-self.agent))
                tagged_tokens = self.tagger.tag_utterance(linked_tokens, self.scenario, self.agent, self.tagged_history)
            # print "Tagged received tokens:", tagged_tokens
            self.update_history(event.agent, tagged_tokens)
            self.update_scores(tagged_tokens)

    def update_history(self, agent_idx, tagged_tokens):
        tokens_without_entities = NgramModel.preprocess_tagged_tokens(tagged_tokens)
        self.tagged_history.append((agent_idx, tagged_tokens))
        self.history.extend(tokens_without_entities)
