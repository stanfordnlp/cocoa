import json

__author__ = 'anushabala'
from src.model.preprocess import markers
from collections import namedtuple
import numpy as np


class Features(object):
    @classmethod
    def name(cls):
        return ''


class MatchFeatures(Features):
    Match = 'MATCH'
    NoMatch = 'NO_MATCH'

    @classmethod
    def name(cls):
        return 'Match'


class MentionFeatures(Features):
    NoMention = 'NO_MENTION'
    Partner = 'PARTNER'
    Me = 'ME'

    @classmethod
    def name(cls):
        return 'Mention'


class SelectionFeatures(Features):
    KnownItem = 'KNOWN_ITEM'
    UnknownItem = 'UNKNOWN_ITEM'

    @classmethod
    def name(cls):
        return "Selection"


MessageFeatures = namedtuple('MessageFeatures', ['Match', 'Mention', 'Selection'])
message_features = MessageFeatures(Match=MatchFeatures(), Mention=MentionFeatures(), Selection=SelectionFeatures())


class Tagger(object):
    SELECTION_TYPE = 'selected_item'

    def __init__(self, type_attribute_mappings):
        """
        Creates a new Tagger object that tags utterances with types and features based on history and KBs.
        The tagger assumes that a lexicon has already identified all tokens in any utterance to be tagged.
        """
        self.type_attribute_mappings = type_attribute_mappings

    def tag_utterance(self, linked_tokens, scenario, agent, tagged_history=[], get_features=True):
        """
        Adds entity type as well as additional history and KB-based features to each token.
        e.g. Given empty tagged history, the linked tokens ['do', 'you', 'know', 'anyone', 'from',
        ('northwestern', ('northwestern university', 'company')]
        would be tagged as
        ['do', 'you', 'know', 'anyone', 'from',
        ('northwestern', ('northwestern university', 'company', ('MATCH', '2')))]

        assuming that the entity 'Northwestern University' matches two items in the current agent's KB.

        :param linked_tokens: The tokens in the utterance to be tagged (with entities identified).
        This function assumes that raw_tokens is a list in the same format as the output of lexicon.link_entity()
        (see lexicon.py).
        :param scenario: The current scenario
        :param agent: Agent index of the current agent
        :param tagged_history: History w.r.t which to tag the utterance. A list of tuples of tagged utterances of the
        form (agent_id, tagged_utterance). If no history is provided, no history-based features are computed.
        :param get_features: Boolean indicating whether to compute history- and KB- based features or not. If False,
        the feature_tuples list in the return value of this function is empty.

        :return: A list of tokens, where every token that matches an entity has the following form:

        (raw_token, (canonical_entity, entity_type, feature_tuples)

        where feature_tuples is a list of any number of tuples, where each tuple corresponds to a single feature.
        The first element in each feature tuple is the name of the tuple (e.g. 'MATCH' or 'MENTION'). Each subsequent element is a string
        comprising the value of that feature. Each feature tuple must have length 2 (the name of the feature
        and its value).

        e.g. ('MATCH', 'NO_MATCH') is a feature of type 'MATCH', and the value of the feature is NO_MATCH. In
        this case, this feature indicates that the entity in question doesn't match any items in the KB of the current
        agent.
        """
        kb = scenario.get_kb(agent)
        tagged_utterance = []
        for token in linked_tokens:
            # if isinstance(token, list):
            #     token = token[0]
            if not isinstance(token, tuple):
                tagged_utterance.append(token)
            else:
                raw, (canonical_entity, entity_type) = token
                if get_features:
                    features = [self.get_message_features(ftype.name(), canonical_entity, entity_type, kb, agent,
                                                          tagged_history)
                                for ftype in message_features]
                    features = [f for f in features if f is not None]
                else:
                    features = []
                tagged_utterance.append((raw, (canonical_entity, entity_type, features)))

        return tagged_utterance

    def tag_selection(self, agent, scenario, preprocessed_tokens):
        # print "In tag_selection"
        # print preprocessed_tokens
        # print str_item
        item = preprocessed_tokens[1]
        kb = scenario.get_kb(agent)
        features = [self.get_selection_features(item, kb)]
        return [markers.SELECT, (item, (item, self.SELECTION_TYPE, features)), markers.EOS]

    def get_selection_features(self, item, kb):
        if item in kb.items:
            return SelectionFeatures.name(), message_features.Selection.KnownItem
        else:
            return SelectionFeatures.name(), message_features.Selection.UnknownItem

    def get_message_features(self, feature_type, canonical_entity, entity_type, kb, agent, tagged_history):
        if feature_type == MatchFeatures.name():
            return self._get_match_features(canonical_entity, entity_type, kb, self.type_attribute_mappings)
        elif feature_type == MentionFeatures.name():
            return self._get_mention_features(canonical_entity, entity_type, agent, tagged_history)
        elif feature_type == SelectionFeatures.name():
            return None
        else:
            raise ValueError('Unsupported feature type: %s')

    @staticmethod
    def _get_match_features(canonical_entity, entity_type, kb, type_attribute_mappings):
        matches = False
        for item in kb.items:
            entity_name = type_attribute_mappings[entity_type]
            # print "Entity name: %s" % entity_name
            if entity_name in item.keys() and item[entity_name].lower() == canonical_entity:
                matches = True
        if matches:
            return MatchFeatures.name(), MatchFeatures.NoMatch
        else:
            return MatchFeatures.name(), MatchFeatures.Match

    @staticmethod
    def _get_mention_features(canonical_entity, entity_type, agent, tagged_history, limit=5):
        i = 0
        for (agent_idx, tagged_message) in reversed(tagged_history):
            for token in reversed(tagged_message):
                if isinstance(token, tuple):
                    # print "in get_mention_features", token
                    _, (tagged_entity, tagged_entity_type, features) = token
                    if tagged_entity == canonical_entity and tagged_entity_type == entity_type:
                        # print "Found mention by agent {} in {}".format(agent_idx, tagged_message)
                        agent_name = MentionFeatures.Me if agent == agent_idx else MentionFeatures.Partner
                        return MentionFeatures.name(), agent_name
            i += 1

            if i == limit:
                break
        return MentionFeatures.name(), MentionFeatures.NoMention