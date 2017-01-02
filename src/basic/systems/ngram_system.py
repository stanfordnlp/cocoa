__author__ = 'anushabala'

from src.basic.systems.system import System
from src.basic.sessions.ngram_session import NgramSession
from src.basic.dataset import Example
import json
from src.basic.ngram_util import preprocess_events
from src.basic.tagger import Tagger
from src.basic.ngram_model import NgramModel
from src.basic.scenario_db import ScenarioDB
from src.model.preprocess import markers
from src.basic.executor import Executor


class NgramSystem(System):
    """
    This class trains an ngram model from a set of training examples
    """
    def __init__(self, transcripts_path, scenarios_path, lexicon, schema, n=7, attribute_specific=True):
        super(NgramSystem, self).__init__()
        transcripts = json.load(open(transcripts_path, 'r'))
        # transcripts = transcripts[:5]
        scenarios = json.load(open(scenarios_path, 'r'))
        self.scenario_db = ScenarioDB.from_dict(schema, scenarios)

        self.lexicon = lexicon
        self.type_attribute_mappings = {v: k for (k, v) in schema.get_attributes().items()}

        self.tagger = Tagger(self.type_attribute_mappings)
        self.n = n
        self.executor = Executor(self.type_attribute_mappings)
        print "Started tagging training data"
        tagged_data, attribute_combos = self.tag_data(transcripts)
        self.models = {}
        print "Training n-gram model for all attribute combinations"
        self.models[NgramSession.DEFAULT_MODEL] = NgramModel(tagged_data, n=self.n)
        if attribute_specific:
            for combo in attribute_combos:
                print "Training n-gram model for attributes {}".format(combo)
                self.models[combo] = NgramModel(tagged_data, n=self.n, attributes=combo)
        print "Trained n-gram model"

    def tag_data(self, raw_data):
        tagged_data = []
        attribute_combos = set()
        i = 0
        for raw_ex in raw_data:
            i += 1
            ex = Example.from_dict(self.scenario_db, raw_ex)
            for agent in [0, 1]:
                attributes = tuple(sorted([attr.name for attr in ex.scenario.attributes]))
                attribute_combos.add(attributes)
                tagged_data.append((self.tag_example(ex, agent), attributes))

            if i % 50 == 0:
                print "Tagged %d examples" % i

        return tagged_data, attribute_combos

    def tag_example(self, example, agent):
        messages = preprocess_events(example.events, agent)
        history = []
        for a_idx, msg in messages:
            if msg[0] == markers.SELECT:
                # selection
                # print "Current agent: %d Utterance agent: %d" % (agent, a_idx)
                # print msg
                tagged_message = self.tagger.tag_selection(agent, example.scenario, msg)
                # print tagged_message
                # print "-----------------------------------------"
            else:
                linked_tokens = self.lexicon.link_entity(msg, uuid=example.uuid)
                # print "Current agent: %d Utterance agent: %d" % (agent, a_idx)
                # print "Raw tokens:", msg
                # print "Linked tokens: ", linked_tokens
                tagged_message = self.tagger.tag_utterance(linked_tokens, example.scenario, agent, history)
                # print tagged_message
                # print "-----------------------------------------"

            history.append((a_idx, tagged_message))

        return history

    def new_session(self, agent, kb, uuid):
        return NgramSession(agent, self.scenario_db.get(uuid), uuid, self.type_attribute_mappings, self.lexicon,
                            self.tagger, self.executor, self.models)

