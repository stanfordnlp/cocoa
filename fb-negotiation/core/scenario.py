from cocoa.core.scenario_db import Scenario as BaseScenario
from cocoa.core.schema import Attribute
from kb import KB

class Scenario(BaseScenario):
    # Agent ids
    FIRST = 0
    SECOND = 1

    def __init__(self, uuid, book, hat, ball, attributes, kbs):
        super(Scenario, self).__init__(uuid, attributes, kbs)
        self.book_count = book
        self.hat_count = hat
        self.ball_count = ball
        # self.bottom_line = 8
        # self.post_id = post_id      // bunch of random numbers: 923461346
        # self.category = category    // phone, housing, bike, furniture, electronics
        # self.images = images        // link to product image: bike/6123601035_0.jpg

    def to_dict(self):
        d = super(Scenario, self).to_dict()
        d['book_count'] = self.book_count
        d['hat_count'] = self.hat_count
        d['ball_count'] = self.ball_count
        return d

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
        book = raw['book_count']
        hat = raw['hat_count']
        ball = raw['ball_count']
        kb_list = [KB.from_dict(scenario_attributes, kb) for kb in raw['kbs']]
        return cls(raw['uuid'], book, hat, ball, scenario_attributes, kb_list)
