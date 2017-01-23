from kb import KB
import numpy as np
from schema import Attribute

def add_scenario_arguments(parser):
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated', required=True)


class Scenario(object):
    '''
    A scenario represents a situation to be played out where each agent has a private KB.
    '''
    def __init__(self, uuid, attributes, kbs, alphas=[]):
        self.uuid = uuid
        self.attributes = attributes
        self.kbs = kbs
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
        return Scenario(raw['uuid'], attributes, [KB.from_dict(attributes, kb) for kb in raw['kbs']], alphas)

    def to_dict(self):
        return {'uuid': self.uuid,
                'attributes': [attr.to_json() for attr in self.attributes],
                'kbs': [kb.to_dict() for kb in self.kbs],
                'alphas': self.alphas}

    def get_kb(self, agent):
        return self.kbs[agent]


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
