from kb import KB
import numpy as np
from schema import Attribute
import src.config as config

def add_scenario_arguments(parser):
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated')

class BaseScenario(object):
    '''
    A scenario represents a situation to be played out where each agent has a KB.
    '''
    def __init__(self, uuid, attributes, kbs):
        self.uuid = uuid
        self.attributes = attributes
        self.kbs = kbs

    @staticmethod
    def from_dict(schema, raw):
        raise NotImplementedError

    def to_dict(self):
        return {'uuid': self.uuid,
                'attributes': [attr.to_json() for attr in self.attributes],
                'kbs': [kb.to_dict() for kb in self.kbs]
                }

    def get_kb(self, agent):
        return self.kbs[agent]

class Scenario(object):
    '''
    Factory of scenarios.
    '''
    @staticmethod
    def get_scenario(*args):
        if config.task == config.MutualFriends:
            return MutualFriendsScenario(*args)
        elif config.task == config.Negotiation:
            return NegotiationScenario(*args)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

    @staticmethod
    def from_dict(schema, raw):
        if config.task == config.MutualFriends:
            return MutualFriendsScenario.from_dict(schema, raw)
        elif config.task == config.Negotiation:
            return NegotiationScenario.from_dict(schema, raw)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

class NegotiationScenario(BaseScenario):
    # Agent ids
    BUYER = 0
    SELLER = 1

    def __init__(self, uuid, post_id, category, images, attributes, kbs):
        super(NegotiationScenario, self).__init__(uuid, attributes, kbs)
        self.post_id = post_id
        self.category = category
        self.images = images

    def to_dict(self):
        d = super(NegotiationScenario, self).to_dict()
        d['post_id'] = self.post_id
        d['category'] = self.category
        return d

    @staticmethod
    def from_dict(schema, raw):
        scenario_attributes = None
        if schema is not None:
            scenario_attributes = schema.attributes
        if 'attributes' in raw.keys():
            scenario_attributes = [Attribute.from_json(a) for a in raw['attributes']]

        if scenario_attributes is None:
            raise ValueError("No scenario attributes found. "
                             "Either schema must not be None (and have valid attributes) or "
                             "scenario dict must have valid attributes field.")
        return NegotiationScenario(raw['uuid'], raw['post_id'], raw['category'], None, scenario_attributes, [KB.from_dict(scenario_attributes, kb) for kb in raw['kbs']])


class MutualFriendsScenario(BaseScenario):
    def __init__(self, uuid, attributes, kbs, alphas=[]):
        super(MutualFriendsScenario, self).__init__(uuid, attributes, kbs)
        self.alphas = alphas

    @staticmethod
    def from_dict(schema, raw):
        alphas = []
        # compatibility with older data format
        if schema is not None:
            attributes = schema.attributes
        else:
            assert 'attributes' in raw
        if 'attributes' in raw:
            attributes = [Attribute.from_json(raw_attr) for raw_attr in raw['attributes']]
        if 'alphas' in raw:
            alphas = raw['alphas']
        return MutualFriendsScenario(raw['uuid'], attributes, [KB.from_dict(attributes, kb) for kb in raw['kbs']], alphas)

    def to_dict(self):
        d = super(MutualFriendsScenario, self).to_dict()
        d['alphas'] = self.alphas
        return d


class ScenarioDB(object):
    '''
    Consists a list of scenarios (specifies the pair of KBs).
    '''
    def __init__(self, scenarios_list):
        self.scenarios_list = scenarios_list  # Keep things in order
        self.size = len(scenarios_list)
        self.scenarios_map = {}  # Map from uuid to scenario
        self.selected_scenarios = set()
        for scenario in scenarios_list:
            self.scenarios_map[scenario.uuid] = scenario

    def get(self, uuid):
        return self.scenarios_map[uuid]

    def select_random(self, exclude_seen=True):
        scenarios = set(self.scenarios_map.keys())

        if exclude_seen:
            scenarios = scenarios - self.selected_scenarios
            if len(scenarios) == 0:
                scenarios = set(self.scenarios_map.keys())
                self.selected_scenarios = set()
        uuid = np.random.choice(list(scenarios))

        return self.scenarios_map[uuid]

    @staticmethod
    def from_dict(schema, raw):
        return ScenarioDB([Scenario.from_dict(schema, s) for s in raw])

    def to_dict(self):
        return [s.to_dict() for s in self.scenarios_list]
