from cocoa.core.scenario_db import Scenario as BaseScenario
from cocoa.core.schema import Attribute
from kb import KB

class Scenario(BaseScenario):
    ## Agent ids
    #FIRST = 0
    #SECOND = 1

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
