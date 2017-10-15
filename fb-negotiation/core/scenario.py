from cocoa.core.scenario_db import Scenario as BaseScenario
from cocoa.core.schema import Attribute
from kb import KB

class Scenario(BaseScenario):
    ## Agent ids
    #FIRST = 0
    #SECOND = 1

    #def __init__(self, uuid, attributes, kbs):
    #    super(Scenario, self).__init__(uuid, attributes, kbs)
    #    # self.bottom_line = 8
    #    # self.post_id = post_id      // bunch of random numbers: 923461346
    #    # self.category = category    // phone, housing, bike, furniture, electronics
    #    # self.images = images        // link to product image: bike/6123601035_0.jpg

    #def to_dict(self):
    #    d = super(Scenario, self).to_dict()
    #    return d

    @classmethod
    def from_dict(cls, schema, raw):
        scenario_attributes = None
        if schema is not None:
            scenario_attributes = schema.attributes
        if 'attributes' in raw.keys():
            scenario_attributes = [Attribute.from_json(a) for a in raw['attributes']]

        if scenario_attributes is None:
            raise ValueError("No scenario attributes found. "
                             "Either schema must not be None (and have valid attributes) or "
                             "scenario dict must have valid attributes field.")
        kb_list = [KB.from_dict(scenario_attributes, kb) for kb in raw['kbs']]
        return cls(raw['uuid'], scenario_attributes, kb_list)
