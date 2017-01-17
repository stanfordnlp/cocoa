'''
Data structures for events, examples, and datasets.
'''

from util import read_json
from event import Event
from scenario_db import Scenario

class Example(object):
    '''
    An example is a dialogue grounded in a scenario, has a set of events, and has some reward at the end.
    Created by through live conversation, serialized, and then read for training.
    '''
    def __init__(self, scenario, uuid, events, outcome, ex_id, agents):
        self.scenario = scenario
        self.uuid = uuid
        self.events = events
        self.outcome = outcome
        self.ex_id = ex_id
        self.agents = agents

    def add_event(self, event):
        self.events.append(event)

    @staticmethod
    def from_dict(scenario_db, raw):
        # compatibility with older data format
        if 'scenario' in raw:
            scenario = Scenario.from_dict(None, raw['scenario'])
        else:
            scenario = scenario_db.get(raw['scenario_uuid'])
        uuid = raw['scenario_uuid']
        events = [Event.from_dict(e) for e in raw['events']]
        outcome = raw['outcome']
        ex_id = raw['uuid']
        if 'agents' in raw:
            agents = raw['agents']
        else:
            agents = None
        return Example(scenario, uuid, events, outcome, ex_id, agents)

    def to_dict(self):
        return {
            'scenario_uuid': self.scenario.uuid,
            'events': [e.to_dict() for e in self.events],
            'outcome': self.outcome,
            'scenario': self.scenario.to_dict(),
            'uuid': self.ex_id,
            'agents': self.agents
        }

class Dataset(object):
    '''
    A dataset consists of a list of train and test examples.
    '''
    def __init__(self, train_examples, test_examples):
        self.train_examples = train_examples
        self.test_examples = test_examples

############################################################

def read_examples(scenario_db, paths, max_examples):
    '''
    Read a maximum of |max_examples| examples from |paths|.
    '''
    examples = []
    for path in paths:
        print 'read_examples: %s' % path
        if max_examples and len(examples) >= max_examples:
            break
        for raw in read_json(path):
            if max_examples and len(examples) >= max_examples:
                break
            examples.append(Example.from_dict(scenario_db, raw))
    return examples

def add_dataset_arguments(parser):
    parser.add_argument('--train-examples-paths', help='Input training examples', nargs='*', default=[])
    parser.add_argument('--test-examples-paths', help='Input test examples', nargs='*', default=[])
    parser.add_argument('--train-max-examples', help='Maximum number of training examples', type=int)
    parser.add_argument('--test-max-examples', help='Maximum number of test examples', type=int)

def read_dataset(scenario_db, args):
    '''
    Return the dataset specified by the given args.
    '''
    train_examples = read_examples(scenario_db, args.train_examples_paths, args.train_max_examples)
    test_examples = read_examples(scenario_db, args.test_examples_paths, args.test_max_examples)
    dataset = Dataset(train_examples, test_examples)
    return dataset
