import pytest
import numpy as np
from cocoa.model.negotiation.preprocess import DialogueBatch, DataGenerator, Preprocessor, create_mappings, TextIntMap, Dialogue
from cocoa.core.schema import Schema
from cocoa.core.negotiation.price_tracker import PriceTracker
from cocoa.core.util import read_json
from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.dataset import read_examples
from itertools import izip

@pytest.fixture(scope='module')
def schema():
    schema_path = 'data/negotiation/craigslist-schema.json'
    return Schema(schema_path)

@pytest.fixture(scope='module')
def scenarios(schema):
    scenarios_path = 'data/negotiation/craigslist-scenarios.json'
    scenario_db = ScenarioDB.from_dict(schema, read_json(scenarios_path))
    return scenario_db

@pytest.fixture(scope='module')
def lexicon():
    return PriceTracker()

@pytest.fixture(scope='module')
def examples(schema):
    data_paths = ['data/negotiation/dev.json']
    return read_examples(None, data_paths, 10)

@pytest.fixture(scope='module')
def preprocessor(schema, lexicon):
    return Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')

@pytest.fixture(scope='module')
def generator(examples, lexicon, schema, preprocessor):
    return DataGenerator(examples, examples, None, preprocessor, schema)

class TestPreprocess(object):
    def test_process_example(self, preprocessor, examples, capsys):
        for dialogue in preprocessor._process_example(examples[0]):
            with capsys.disabled():
                print '\n========== Example dialogue (speaking agent=%d) ==========' % dialogue.agent
                print examples[0]
                for i, (agent, turn) in enumerate(izip(dialogue.agents, dialogue.token_turns)):
                    print 'agent=%d' % agent
                    for utterance in turn:
                        print utterance

    @pytest.fixture(scope='class')
    def processed_dialogues(self, preprocessor, examples):
        dialogues = [dialogue for dialogue in preprocessor._process_example(examples[0])]
        return dialogues

    @pytest.fixture(scope='class')
    def textint_map(self, processed_dialogues, schema, preprocessor):
        mappings = create_mappings(processed_dialogues, schema, preprocessor.entity_forms.values())
        textint_map = TextIntMap(mappings['vocab'], preprocessor)
        return textint_map

    def test_price_hist(self, processed_dialogues, textint_map, capsys):
        dialogue = processed_dialogues[0]
        dialogue.entities = dialogue._flatten_turns(dialogue.entities, None)
        from cocoa.model.negotiation.preprocess import markers, SpecialSymbols
        price_hists = dialogue.get_price_hist(3, -100)

        with capsys.disabled():
            for turn, entity_turn, price_turn in izip(dialogue.token_turns, dialogue.entities, price_hists):
                print 'utterance:', turn
                print 'entities:', entity_turn
                print 'price hist:', price_turn

    def test_normalize_turn(self, generator, capsys):
        generator.convert_to_int()
        dialogues = generator.dialogues['train'][:2]
        batches = generator.create_dialogue_batches(dialogues, 1)
        assert len(batches) == 2

        with capsys.disabled():
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            for i in xrange(2):  # Which perspective
                batch = batches[i]
                print '\n========== Example batch =========='
                for j in xrange(1):  # Which batch
                    print 'agent:', batch['agent'][j]
                    batch['kb'][j].dump()
                    for t, b in enumerate(batch['batch_seq']):
                        print t
                        print 'encode:', generator.textint_map.int_to_text(b['encoder_inputs'][j])
                        print 'encoder tokens:', None if not b['encoder_tokens'] else b['encoder_tokens'][j]
                        print 'decode:', generator.textint_map.int_to_text(b['decoder_inputs'][j])
                        print 'price:'
                        print b['decoder_price_inputs']
                        print 'targets:', generator.textint_map.int_to_text(b['targets'][j])
                        print 'price targets:', b['price_targets']
                        print 'decoder tokens:', b['decoder_tokens'][j]

