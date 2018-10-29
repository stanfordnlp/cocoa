from cocoa.core.scenario_db import Scenario as BaseScenario
from cocoa.core.schema import Attribute
from kb import KB

class Scenario(BaseScenario):
    # Agent ids
    BUYER = 0
    SELLER = 1

    def __init__(self, uuid, post_id, category, images, attributes, kbs):
        super(Scenario, self).__init__(uuid, attributes, kbs)
        self.post_id = post_id
        self.category = category
        self.images = images

    def to_dict(self):
        d = super(Scenario, self).to_dict()
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
        return Scenario(raw['uuid'], raw['post_id'], raw['category'], None, scenario_attributes, [KB.from_dict(scenario_attributes, kb) for kb in raw['kbs']])
