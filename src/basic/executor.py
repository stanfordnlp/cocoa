import json

__author__ = 'anushabala'
from src.basic.tagger import MentionFeatures, MatchFeatures, SelectionFeatures
from src.model.preprocess import markers


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


