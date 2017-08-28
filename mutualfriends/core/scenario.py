from cocoa.core.scenario_db import Scenario as BaseScenario
from cocoa.core.schema import Attribute
from kb import KB

class Scenario(BaseScenario):
    def __init__(self, uuid, attributes, kbs, alphas=[]):
        super(Scenario, self).__init__(uuid, attributes, kbs)
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
        d = super(Scenario, self).to_dict()
        d['alphas'] = self.alphas
        return d
