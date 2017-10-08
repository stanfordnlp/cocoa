import random

from cocoa.core.util import read_json, write_json

from utils import xml_safe

def add_eval_data_arguments(parser):
    parser.add_argument('--system-outputs', nargs='*', help='Path to JSON data or raw generation results in forms of (system, path, ...)')
    parser.add_argument('--eval-data', help='Path to saved evaluation data in JSON')
    parser.add_argument('--eval-data-output', help='Path to write evaluation data in JSON')
    parser.add_argument('--num-context', default=3, help='Number of utterances to display in the context')

class EvalData(object):
    def __init__(self, data):
        self.data = data

    def dump(self, output):
        print 'Dumping data to {}'.format(output)
        write_json(self.data, output)

    def sample_examples(self, num_context, evaluated=set(), systems=None, pairs=None, qids=None):
        """Randomly sample num_context examples to evaluate.

        Args:
            num_context (int)
            evaluated (set): set of example ids that have been evaluated
            systems (list): only use responses from `systems`
            pairs (list): return pairs of responses for CompareTask
            qids (list): list of example ids to use

        Returns:
            examples (list[(qid, context, response)]): tuples that can be used to contruct questions by EvalTask.

        """
        if not qids:
            example_ids = [x for x in self.data.keys() if not x in evaluated]
        else:
            example_ids = qids
        if num_context > len(example_ids):
            print 'WARNING: wanted {} examples, only has {}'.format(num_context, len(example_ids))
            num_context = len(example_ids)
        selected_ids = random.sample(example_ids, num_context)
        examples = []
        for id_ in selected_ids:
            responses = self.data[id_]['responses']
            context = self.data[id_]['context']
            if not pairs:
                for system, response in responses.iteritems():
                    if systems is not None and not system in systems:
                        continue
                    qid = '{system}-{ex_id}'.format(system=system, ex_id=id_)
                    examples.append((qid, context, response))
            else:
                qid = '{system0}-{system1}-{ex_id}'.format(system0=pairs[0], system1=pairs[1], ex_id=id_)
                responses = [responses[system] for system in pairs]
                examples.append((qid, context, responses))
        return examples, selected_ids

    @classmethod
    def valid_example(cls, example, num_context_utterances):
        """Filter some examples that should not be evaluated.

        Args:
            example (dict): an entry in system outputs (see read_system_responses)
            num_context_utterances (int)

        Returns:
            valid (bool)

        """
        if len(example['prev_turns']) < num_context_utterances:
            return False
        return True

    @classmethod
    def read_system_responses(cls, system, path, num_context_utterances, data):
        """Read system responses and update the database.

        Args:
            system (str): system name
            path (str): a JSON file containing system outputs.
                |-[]
                 |-ex_id (str): unique id that identifies a context-reference pair
                 |-prev_turns (list)
                 |-reference
                 |-response
            num_context_utterances (int)
            data (dict): see from_file.

        """
        examples = read_json(path)
        for ex in examples:
            if not cls.valid_example(ex, num_context_utterances):
                continue
            qid = ex['ex_id']
            context_turns = ex['prev_turns'][-1*num_context_utterances:]
            agent_names = cls.get_agent_name(context_turns + [ex['reference']])
            context = []
            for i, u in enumerate(context_turns):
                u = cls.process_utterance(u, role=agent_names[i])
                if len(u[1]) > 0:
                    context.append(u)
            reference = cls.process_utterance(ex['reference'], role=agent_names[-1])
            response = cls.process_utterance(ex['response'], role=agent_names[-1])
            if not (len(reference) and len(response) and len(context)):
                continue
            if qid not in data:
                data[qid] = {
                        'context': context,
                        'responses': {}
                        }
            assert system not in data[qid]['responses']
            data[qid]['responses'][system] = response
            data[qid]['responses']['reference'] = reference

    @classmethod
    def get_agent_name(cls, turns):
        num_turns = len(turns)
        return ['A' if i % 2 == 0 else 'B' for i in num_turns]

    @classmethod
    def process_utterance(cls, utterance, role=''):
        """Process the utterance to a form to be displayed to turkers.

        Returns:
            agent, utterance: (str, str)

        """
        return (role, utterance)

    @classmethod
    def from_json(cls, data_path):
        """Construct from dumped data.
        """
        data = read_json(data_path)
        return cls(data)

    @classmethod
    def from_file(cls, paths, num_context_utterances=3):
        """Read context-response pairs from system outputs.

        Args:
            paths (list[tuple]): [(system, path)]
            num_context_utterances (int): number of utterances to use as context

        Returns:
            data (dict):
                |-qid (str)
                 |-context ([(agent, utterance)])
                 |-responses
                  |-system (str)
                   |-response ((agent, utterance))

        """
        data = {}
        for system, path in paths:
            cls.read_system_responses(system, path, num_context_utterances, data)
        return cls(data)
