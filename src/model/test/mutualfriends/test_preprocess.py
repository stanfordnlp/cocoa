import pytest
from itertools import izip
from model.preprocess import DialogueBatch, DataGenerator
from basic.dataset import read_examples
from basic.schema import Schema
from basic.util import read_json
from basic.lexicon import Lexicon
from basic.scenario_db import ScenarioDB
from model.graph import GraphMetadata, Graph
import numpy as np
from numpy.testing import assert_array_equal

class TestPreprocess(object):
    @pytest.fixture(scope='session')
    def schema(self):
        schema_path = 'data/friends-schema.json'
        return Schema(schema_path)

    @pytest.fixture(scope='session')
    def examples(self, schema):
        scenarios_path = 'output/friends-scenarios.json'
        data_paths = ['output/friends-train-examples.json']
        scenario_db = ScenarioDB.from_dict(schema, read_json(scenarios_path))
        return read_examples(scenario_db, data_paths, 10)

    @pytest.fixture(scope='session')
    def lexicon(self, schema):
        return Lexicon(schema, learned_lex=False)

    @pytest.fixture(scope='class')
    def generator(self, examples, lexicon, schema):
        return DataGenerator(examples, examples, None, schema, lexicon, 5, use_kb=True)

    def set_metadata(self, schema, generator):
        Graph.metadata = GraphMetadata(schema, generator.entity_map, generator.relation_map, 4, 10)

    def test_preprocess(self, generator, examples, capsys):
        dialogue = generator._process_example(examples[0])
        int_dialogue = generator.dialogues['train'][0]
        with capsys.disabled():
            print '\n========== Example dialogu by turns =========='
            for i, (agent, turn) in enumerate(izip(dialogue.agents, dialogue.turns[0])):
                print 'agent=%d' % agent
                for utterance in turn:
                    print utterance
                print 'Integers:'
                for utterance in int_dialogue.turns[0][i]:
                    print utterance
                    print map(generator.vocab.to_word, utterance)

    @pytest.fixture(scope='session')
    def dialogue_batch(self, generator):
        dialogues = generator.dialogues['train'][:2]
        return  DialogueBatch(dialogues)

    @pytest.fixture(scope='session')
    def test_dialogue_batch(self, generator):
        dialogues = generator.dialogues['dev'][:2]
        return  DialogueBatch(dialogues)

    def test_normalize_dialogue(self, generator, dialogue_batch, capsys):
        dialogue_batch._normalize_dialogue()
        assert len(dialogue_batch.dialogues[0].turns[0]) == len(dialogue_batch.dialogues[1].turns[0])
        with capsys.disabled():
            print '\n========== Example flattened turn =========='
            turn = dialogue_batch.dialogues[0].turns[0][0]
            print turn
            print map(generator.vocab.to_word, turn)

    def test_create_turn_batches(self, generator, dialogue_batch, capsys):
        turn_batches = dialogue_batch._create_turn_batches()[0]
        with capsys.disabled():
            print '\n========== Example turn batch =========='
            for i in xrange(2):
                print turn_batches[i].shape
                print turn_batches[i]
                print map(generator.vocab.to_word, list(turn_batches[i][0]))
                print map(generator.vocab.to_word, list(turn_batches[i][1]))

    def test_create_batches(self, generator, test_dialogue_batch, capsys):
        batches = test_dialogue_batch.create_batches()
        with capsys.disabled():
            for i in xrange(2):  # Which perspective
                batch = batches[i]
                print '\n========== Example batch =========='
                for j in xrange(1):  # Which batch
                    print 'agent:', batch['agent'][j]
                    batch['kb'][j].dump()
                    for t, b in enumerate(batch['batch_seq']):
                        print t
                        print 'encode:', map(generator.vocab.to_word, list(b['encoder_inputs'][j]))
                        print 'encode last ind:', b['encoder_inputs_last_inds'][j]
                        print 'encoder tokens:', None if not b['encoder_tokens'] else b['encoder_tokens'][j]
                        print 'decode:', map(generator.vocab.to_word, list(b['decoder_inputs'][j]))
                        print 'decode last ind:', b['decoder_inputs_last_inds'][j]
                        print 'targets:', map(generator.vocab.to_word, list(b['targets'][j]))
                        print 'decoder tokens:', b['decoder_tokens'][j]

    def test_generator(self, schema, generator):
        self.set_metadata(schema, generator)
        batch_size = 4
        generator = generator.generator('train', batch_size)
        num_batches = generator.next()
        batch = generator.next()
        #assert len(batch['agent']) == batch_size
        #assert len(batch['kb']) == batch_size
        #assert batch['batch_seq'][0]['decoder_inputs'].shape[0] == batch_size

    def test_last_inds(self, dialogue_batch):
        pad = -1
        inputs = np.array([[1,2,3],
                           [1,pad, pad]], dtype=np.int32)
        inds = dialogue_batch._get_last_inds(inputs, pad)
        expected = np.array([2, 0])
        assert_array_equal(inds, expected)
