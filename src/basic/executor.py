import json

__author__ = 'anushabala'
from src.basic.tagger import MentionFeatures, MatchFeatures, SelectionFeatures
from src.model.preprocess import markers, dict_item_to_entity
from collections import defaultdict
from src.basic.tagger import Tagger
from src.model.vocab import is_entity
import numpy as np


class Executor(object):
    def __init__(self, type_attribute_mappings):
        self.type_attribute_mappings = type_attribute_mappings

    def get_matched_entities(self, entity_type, kb):
        entities = []
        entity_name = self.type_attribute_mappings[entity_type]
        for item in kb.items:
            if entity_name not in item.keys():
                return []
            name = item[entity_name]
            entities.append(name)

        return entities

    def get_mentioned_entities(self, entity_type, current_agent, agent_name, history, limit=5):
        entities = []
        if agent_name == MentionFeatures.NoMention:
            return entities

        mentioning_agent_id = current_agent if agent_name == MentionFeatures.Me else 1 - current_agent
        i = 0
        for (agent_idx, tagged_message) in reversed(history):
            if agent_idx != mentioning_agent_id:
                continue
            for token in reversed(tagged_message):
                if isinstance(token, tuple):
                    raw, (canonical, found_type, _ ) = token
                    if found_type == entity_type:
                        entities.append(canonical)
            i += 1
            if i == limit:
                break

        return entities

    def get_selection_candidates(self, agent, kb, history):
        for agent_idx, tagged_message in reversed(history):
            if agent_idx != agent and tagged_message[0] == markers.SELECT:
                # print "in get_selected_item", tagged_message
                item, (_, _, selection_features) = tagged_message[1]
                if selection_features[0][1] == SelectionFeatures.KnownItem:
                    return [item]

        # else return all possible candidates (all items in KB)
        return kb.items

    def get_candidate_entities(self, tagged_token, agent, scenario, tagged_history=[], kb=None):
        """
        Returns a candidate set of entities that can be generated given the features for the current token and a
        tagged history of the dialogue so far.
        :param tagged_token: Tuple of form (entity_type, feature_tuples) where feature_tuples is a list of feature
        tuples, where each feature tuple contains the feature name and value (see the Tagger class in tagger.py).
        :param agent: Agent index of the current agent
        :param scenario: Current scenario (instance of the Scenario class)
        :param tagged_history: Tagged history of the dialogue so far: a list of tuples of tagged utterances of the
        form (agent_id, tagged_utterance). If no history is provided, no entities will be found for history-based
        features (e.g. mention features).
        :return: A list of candidate entities based on the entity type and features.
        """
        if not kb:
            kb = scenario.get_kb(agent)
        entity_type, features = tagged_token

        found_features = {feature_tuple[0]: feature_tuple[1] for feature_tuple in features}
        if SelectionFeatures.name() in found_features.keys():
            return self.get_selection_candidates(agent, kb, tagged_history)

        matched_entities = self.get_matched_entities(entity_type, kb)

        agent_name = found_features[MentionFeatures.name()]
        mentioned_entities = self.get_mentioned_entities(entity_type, agent, agent_name, tagged_history)

        if found_features[MatchFeatures.name()] == MatchFeatures.Match and found_features[MentionFeatures.name()] != MentionFeatures.NoMention:
            # Mention and match features found
            choices = list(set(mentioned_entities).intersection(set(matched_entities)))

        elif found_features[MatchFeatures.name()] == MatchFeatures.Match:
            # No mention features
            choices = matched_entities

        elif found_features[MentionFeatures.name()] != MentionFeatures.NoMention:
            choices = mentioned_entities

        else:
            # No mention or match features
            # This should never happen (never generate an entity that isn't in the KB and wasn't previously mentioned
            choices = []

        if len(choices) == 0:
            # If any candidate sets are empty, return None and try to regenerate
            return None

        return choices

    def new_session_executor(self, kb, random=True):
        return SessionExecutor(self.type_attribute_mappings, kb, random=random)

class SessionExecutor(Executor):
    def __init__(self, type_attribute_mappings, kb, random=True):
        super(SessionExecutor, self).__init__(type_attribute_mappings)
        self.kb = kb
        self.entity_scores = defaultdict(lambda : defaultdict(int))
        for item in self.kb.items:
            for attr_name, attr_value in item.iteritems():
                self.entity_scores[attr_name][attr_value] = 1.
        self.random = random

    def execute(self, tagged_tokens, agent, tagged_history):
        executed_tokens = []
        for token in tagged_tokens:
            if not is_entity(token):
                executed_tokens.append(token)
            else:
                entity_type, features = token
                candidates = self.get_candidate_entities(token, agent, None, tagged_history, kb=self.kb)
                if not candidates:
                    if not self.is_token_valid(token):
                        # if entity type isn't valid return none
                        # print "[Agent %d]  Invalid token: " % self.agent, token
                        candidates = None
                    else:
                        # if features lead to invalid state just choose best ranked entity of this type
                        # print "[Agent %d]  Invalid state from features:" % self.agent, token
                        if entity_type == Tagger.SELECTION_TYPE:
                            candidates = self.kb.items
                        else:
                            attr_name = self.type_attribute_mappings[entity_type]
                            candidates = [x[attr_name] for x in self.kb.items]
                if candidates is None:
                    executed_tokens.append('')
                else:
                    if not self.random:
                        best_candidate, score = self.select_candidate_from_type(entity_type, candidates)
                    else:
                        best_candidate = np.random.choice(candidates)
                    if entity_type == Tagger.SELECTION_TYPE:
                        executed_tokens.append(dict_item_to_entity(self.kb, best_candidate)[1])
                    else:
                        executed_tokens.append((best_candidate, entity_type))
        return executed_tokens


    def is_token_valid(self, token):
        if is_entity(token):
            entity_type, features = token
            return entity_type in [attr.value_type for attr in self.kb.attributes] or entity_type == Tagger.SELECTION_TYPE
        return True

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

    def update_mentioned(self, entity_tokens):
        for token in entity_tokens:
            if is_entity(token):
                entity_name, entity_type = token
                if entity_type != 'item':
                    attr_name = self.type_attribute_mappings[entity_type]
                    self.entity_scores[attr_name][entity_name] = 0.

    def select_candidate_from_type(self, entity_type, candidates):
        if entity_type == Tagger.SELECTION_TYPE:
            return self.select_best_item(candidates)
        attr_name = self.type_attribute_mappings[entity_type]
        candidate_scores = [(c, self.entity_scores[attr_name][c]) for c in candidates]
        # print "[Agent %d] Sorted candidates (type %s):" % (self.agent, entity_type)
        # print list(sorted(candidate_scores, key=lambda x: x[1], reverse=True))
        return max(candidate_scores, key=lambda x: x[1])

    def select_best_item(self, candidates):
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


