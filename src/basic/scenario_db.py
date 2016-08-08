import json
from kb import KB

def add_scenario_arguments(parser):
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated', required=True)


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

    @staticmethod
    def from_dict(schema, raw):
        return ScenarioDB([Scenario.from_dict(schema, s) for s in raw])
    def to_dict(self):
        return [s.to_dict() for s in self.scenarios_list]
