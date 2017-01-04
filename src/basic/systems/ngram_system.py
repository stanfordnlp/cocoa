__author__ = 'anushabala'

from src.basic.systems.system import System
from src.basic.sessions.ngram_session import NgramSession
import json
from src.basic.tagger import Tagger, DatasetTagger
from src.basic.ngram_model import NgramModel
from src.basic.scenario_db import ScenarioDB
from src.basic.executor import Executor


class NgramSystem(System):
    """
    This class trains an ngram model from a set of training examples
    """
    def __init__(self, transcripts_path, scenarios_path, lexicon, schema, n=7, attribute_specific=True):
        super(NgramSystem, self).__init__()
        transcripts = json.load(open(transcripts_path, 'r'))
        transcripts = transcripts[:100]
        scenarios = json.load(open(scenarios_path, 'r'))
        print 'Creating new NgramSystem: transcripts=%s, scenarios=%s, n=%d' % (transcripts_path, scenarios_path, n)
        self.scenario_db = ScenarioDB.from_dict(schema, scenarios)

        self.lexicon = lexicon
        self.type_attribute_mappings = {v: k for (k, v) in schema.get_attributes().items()}

        self.tagger = Tagger(self.type_attribute_mappings)
        self.dataset_tagger = DatasetTagger(lexicon, self.tagger, self.scenario_db)

        self.n = n
        self.executor = Executor(self.type_attribute_mappings)
        print "[NgramSystem] Started tagging training data"
        tagged_data, attribute_combos = self.dataset_tagger.tag_data(transcripts)
        self.models = {}
        print "[NgramSystem] Training n-gram model for all attribute combinations"
        self.models[NgramSession.DEFAULT_MODEL] = NgramModel(tagged_data, n=self.n)
        if attribute_specific:
            for combo in attribute_combos:
                print "[NgramSystem] Training n-gram model for attributes {}".format(combo)
                self.models[combo] = NgramModel(tagged_data, n=self.n, attributes=combo)
        print "[NgramSystem] Trained n-gram model"

    def new_session(self, agent, kb, uuid):
        return NgramSession(agent, self.scenario_db.get(uuid), uuid, self.type_attribute_mappings, self.lexicon,
                            self.tagger, self.executor, self.models)

    @classmethod
    def name(cls):
        return 'ngram'

