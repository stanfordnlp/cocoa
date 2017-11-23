'''
Data structures for events, examples, and datasets.
'''

from util import read_json
from event import Event
from kb import KB

class Example(object):
    '''
    An example is a dialogue grounded in a scenario, has a set of events, and has some reward at the end.
    Created by through live conversation, serialized, and then read for training.
    '''
    def __init__(self, scenario, uuid, events, outcome, ex_id, agents, agents_info=None):
        self.scenario = scenario
        self.uuid = uuid
        self.events = events
        self.outcome = outcome
        self.ex_id = ex_id
        self.agents = agents
        self.agents_info = agents_info

    def add_event(self, event):
        self.events.append(event)

    @classmethod
    def from_dict(cls, raw, Scenario, scenario_db=None):
        if 'scenario' in raw:
            scenario = Scenario.from_dict(None, raw['scenario'])
        # Compatible with old data formats (to be removed)
        elif scenario_db:
            print 'WARNING: scenario should be provided in the example'
            scenario = scenario_db.get(raw['scenario_uuid'])
        else:
            raise ValueError('No scenario')
        uuid = raw['scenario_uuid']
        events = Event.gather_eval([Event.from_dict(e) for e in raw['events']])
        outcome = raw['outcome']
        ex_id = raw['uuid']
        if 'agents' in raw:
            agents = {int(k): v for k, v in raw['agents'].iteritems()}
        else:
            agents = None
        agents_info = raw.get('agents_info', None)
        return Example(scenario, uuid, events, outcome, ex_id, agents, agents_info=agents_info)

    @classmethod
    def test_dict(cls, raw):
        uuid = raw['scenario_uuid']
        events = Event.gather_eval([Event.from_dict(e) for e in raw['events']])
        outcome = raw['outcome']
        ex_id = raw['uuid']
        if 'agents' in raw:
            agents = {int(k): v for k, v in raw['agents'].iteritems()}
        else:
            agents = None
        agents_info = raw.get('agents_info', None)
        return Example(None, uuid, events, outcome, ex_id, agents, agents_info=agents_info)


    def to_dict(self):
        return {
            'scenario_uuid': self.scenario.uuid,
            'events': [e.to_dict() for e in self.events],
            'outcome': self.outcome,
            'scenario': self.scenario.to_dict(),
            'uuid': self.ex_id,
            'agents': self.agents,
            'agents_info': self.agents_info,
        }

class Dataset(object):
    '''
    A dataset consists of a list of train and test examples.
    '''
    def __init__(self, train_examples, test_examples):
        self.train_examples = train_examples
        self.test_examples = test_examples

class EvalExample(object):
    '''
    Context-response pairs with scores from turkes.
    '''
    def __init__(self, uuid, kb, agent, role, prev_turns, prev_roles, target, candidates, scores):
        self.ex_id = uuid
        self.kb = kb
        self.agent = agent
        self.role = role
        self.prev_turns = prev_turns
        self.prev_roles = prev_roles
        self.target = target
        self.candidates = candidates
        self.scores = scores

    @staticmethod
    def from_dict(schema, raw):
        ex_id = raw['exid']
        kb = KB.from_dict(schema.attributes, raw['kb'])
        agent = raw['agent']
        role = raw['role']
        prev_turns = raw['prev_turns']
        prev_roles = raw['prev_roles']
        target = raw['target']
        candidates = raw['candidates']
        scores = raw['results']
        return EvalExample(ex_id, kb, agent, role, prev_turns, prev_roles, target, candidates, scores)

############################################################

def read_examples(paths, max_examples, Scenario):
    '''
    Read a maximum of |max_examples| examples from |paths|.
    '''
    examples = []
    for path in paths:
        print 'read_examples: %s' % path
        for raw in read_json(path):
            if max_examples >= 0 and len(examples) >= max_examples:
                break
            examples.append(Example.from_dict(raw, Scenario))
    return examples

def add_dataset_arguments(parser):
    parser.add_argument('--train-examples-paths', nargs='*', default=[],
        help='Input training examples')
    parser.add_argument('--test-examples-paths', nargs='*', default=[],
        help='Input test examples')
    parser.add_argument('--train-max-examples', type=int,
        help='Maximum number of training examples')
    parser.add_argument('--test-max-examples', type=int,
        help='Maximum number of test examples')
    parser.add_argument('--eval-examples-paths', nargs='*', default=[],
        help='Path to multi-response evaluation files')

def read_dataset(args, Scenario):
    '''
    Return the dataset specified by the given args.
    '''
    train_examples = read_examples(args.train_examples_paths, args.train_max_examples, Scenario)
    test_examples = read_examples(args.test_examples_paths, args.test_max_examples, Scenario)
    dataset = Dataset(train_examples, test_examples)
    return dataset

if __name__ == "__main__":
    raw = read_json("fb-negotiation/data/transformed_test.json")
    for idx, example in enumerate(raw):
        print Example.test_dict(example)
