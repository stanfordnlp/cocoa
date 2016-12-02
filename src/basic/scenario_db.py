from kb import KB
import numpy as np

def add_scenario_arguments(parser):
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--scenarios-path', nargs='*', default=[], help='Output path for the scenarios generated', required=True)


class Scenario(object):
    '''
    A scenario represents a situation to be played out where each agent has a private KB.
    '''
    def __init__(self, uuid, kbs):
        self.uuid = uuid
        self.kbs = kbs

    @staticmethod
    def from_dict(schema, raw):
        return Scenario(raw['uuid'], [KB.from_dict(schema, kb) for kb in raw['kbs']])
    def to_dict(self):
        return {'uuid': self.uuid, 'kbs': [kb.to_dict() for kb in self.kbs]}

    def get_kb(self, agent):
        return self.kbs[agent]


class ScenarioDB(object):
    '''
    Consists a list of scenarios (specifies the pair of KBs).
    '''
    def __init__(self, scenarios_list):
        self.scenarios_list = scenarios_list  # Keep things in order
        self.scenarios_map = {}  # Map from uuid to scenario
        for scenario in scenarios_list:
            self.scenarios_map[scenario.uuid] = scenario

    def get(self, uuid):
        return self.scenarios_map[uuid]

    def select_random(self, exclude_set=None):
        scenarios = set(self.scenarios_map.keys())
        if exclude_set:
            scenarios = scenarios - exclude_set
            if len(scenarios) == 0:
                scenarios = set(self.scenarios_map.keys())
        uuid = np.random.choice(scenarios)
        return self.scenarios_map[uuid]

    @staticmethod
    def from_dict(schema, raw_list):
        return ScenarioDB([Scenario.from_dict(schema, s) for raw in raw_list for s in raw])
    def to_dict(self):
        return [s.to_dict() for s in self.scenarios_list]
