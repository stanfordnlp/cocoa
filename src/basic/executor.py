import json

__author__ = 'anushabala'
from src.basic.tagger import MentionFeatures, MatchFeatures, SelectionFeatures
from src.model.preprocess import markers
import numpy as np


class Executor(object):
    def __init__(self, type_attribute_mappings):
        self.type_attribute_mappings = type_attribute_mappings

    def get_matched_entities(self, entity_type, kb):
        entities = set()
        entity_name = self.type_attribute_mappings[entity_type]
        for item in kb.items:
            if entity_name not in item.keys():
                return []
            name = item[entity_name]
            entities.add(name)

        return list(entities)

    def get_mentioned_entities(self, entity_type, current_agent, agent_name, history):
        entities = {}
        mentioning_agent_id = current_agent if agent_name == MentionFeatures.Mention else 1 - current_agent
        for (agent_idx, tagged_message) in reversed(history):
            if agent_idx != mentioning_agent_id:
                continue
            for token in reversed(tagged_message):
                if isinstance(token, tuple):
                    raw, (canonical, found_type, _ ) = token
                    if found_type == entity_type:
                        entities[canonical] = raw

        return entities

    def get_selected_item(self, agent, kb, history):
        for agent_idx, tagged_message in reversed(history):
            if agent_idx != agent and tagged_message[0] == markers.SELECT:
                # print "in get_selected_item", tagged_message
                str_item, (_, _, selection_features) = tagged_message[1]
                if selection_features[0][1] == SelectionFeatures.KnownItem:
                    return str_item, str_item

        # else select random item from KB
        idxes = np.arange(len(kb.items))
        str_item = json.dumps(kb.items[np.random.choice(idxes)])
        return str_item, str_item

    def get_entity(self, tagged_token, agent, scenario, tagged_history=[]):
        # print "In get_entity"
        kb = scenario.get_kb(agent)
        entity_type, features = tagged_token
        # print "entity type", entity_type
        # print "features", features
        matched_entities = {}
        mentioned_entities = {}
        for feature_tuple in features:
            feature_name = feature_tuple[0]
            if feature_name == SelectionFeatures.name():
                return self.get_selected_item(agent, kb, tagged_history)

            if feature_name == MatchFeatures.name():
                val = feature_tuple[1]
                if val == MatchFeatures.Match:
                    matched_entities = self.get_matched_entities(entity_type, kb)

            if feature_name == MentionFeatures.name():
                agent_name = feature_tuple[2]
                mentioned_entities = self.get_mentioned_entities(entity_type, agent, agent_name, tagged_history)

        choices = self.get_matched_entities(entity_type, kb)
        if len(mentioned_entities.keys()) > 0 and len(matched_entities) > 0:
            # try to find intersection
            intersection = set(mentioned_entities.keys()).intersection(set(matched_entities))
            if len(intersection) > 0:
                choices = list(intersection)
            else:
                choices = matched_entities

        if len(mentioned_entities.keys()) > 0 and len(matched_entities) == 0:
            choices = mentioned_entities.keys()

        if len(matched_entities) > 0 and len(mentioned_entities.keys()) == 0:
            choices = matched_entities

        # print choices
        if len(choices) == 0:
            return None, None
        idxes = np.arange(len(choices))
        i = np.random.choice(idxes)
        # todo use entityrealizer to return raw and canonical entities
        chosen = choices[i]
        return chosen, chosen

