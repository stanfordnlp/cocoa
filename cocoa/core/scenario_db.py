class Scenario(object):
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


class ScenarioDB(object):
    '''
    Consists a list of scenarios (specifies the pair of KBs).
    '''
    def __init__(self, scenarios_list):
        self.scenarios_list = scenarios_list  # Keep things in order
        self.scenarios_map = {}  # Map from uuid to scenario
        self.selected_scenarios = set()
        for scenario in scenarios_list:
            self.scenarios_map[scenario.uuid] = scenario
        self.size = len(self.scenarios_map)

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
    def from_dict(schema, raw, scenario_class):
        return ScenarioDB([scenario_class.from_dict(schema, s) for s in raw])

    def to_dict(self):
        return [s.to_dict() for s in self.scenarios_list]
