import random
import string
from argparse import ArgumentParser

from cocoa.core.util import read_json, write_json
from cocoa.turk.utils import get_mturk_connection, xml_safe
from cocoa.turk.task import EvalTask

class NegotiationEvalData(object):
    def __init__(self, path, num_context_utterances=3):
        eval_examples = read_json(path)
        eval_data = []
        for example in eval_examples:
            if len(example['prev_turns']) < num_context_utterances:
                continue
            last_utterance = example['prev_turns'][-1]
            if '<offer>' in last_utterance or '<accept>' in last_utterance or '<reject>' in last_utterance:
                continue
            if example['candidates'] is None or len(example['candidates']) < 5:
                continue
            # Pick dialogues in the middle
            if example['turn_id'] < 2:
                continue

            context = []
            for u in example['prev_turns'][-1*num_context_utterances:]:
                u = self.process_utterance(u)
                if len(u[1]) > 0:
                    context.append(u)
            target = self.process_utterance(example['target'])
            retrieved = self.process_utterance(example['candidates'][0]['response'])
            ex_id = example['exid']

            eval_data.append((
                ('target-{}'.format(ex_id), context, target),
                ('retrieved-{}'.format(ex_id), context, retrieved),
                ))

        self.eval_data = eval_data
        random.shuffle(self.eval_data)

    def iter(self, num_unique_context=None, num_evals_per_instance=None, num_instances_per_batch=None):
        if num_unique_context > len(self.eval_data):
            print 'WARNING: wanted {} but only has {}'.format(num_unique_context , len(self.eval_data))
            sys.exit()

        data = self.eval_data[:num_unique_context]
        # NOTE: each datum contains a target instance and a retrieved instance
        # Mix them now.
        data = [x for d in data for x in d]
        random.shuffle(data)

        N = len(data)
        for i in xrange(0, N * num_evals_per_instance, num_instances_per_batch):
            yield [data[n % N] for n in xrange(i, i + num_instances_per_batch)]

    def process_utterance(self, utterance):
        if utterance[0] == '<go-b>':
            role = 'buyer'
        else:
            role = 'seller'
        tokens = []
        for w in utterance:
            if not isinstance(w, basestring):
                if w[1][1] == 'price':
                    tokens.append('PRICE')
                else:
                    raise ValueError
            elif w in ('<offer>', '<accept>', '<reject>'):
                tokens.append(w[1:-1].upper())
            elif len(w) > 2 and w[0] == '<' and w[-1] == '>':
                continue
            elif (w in string.punctuation or "'" in w) and len(tokens) > 0:
                tokens[-1] += w
            else:
                tokens.append(w)
        return (role, xml_safe(' '.join(tokens)))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--review', action='store_true', help='Review completed hits')
    parser.add_argument('--debug', action='store_true', help='If ture, run in sandbox')
    parser.add_argument('--aws-config', required=True, help='AWS credential')
    args = parser.parse_args()

    data = NegotiationEvalData('data/dev_candidates.json')

    config = read_json(args.aws_config)
    mtc = get_mturk_connection(config, debug=args.debug)
    task = EvalTask(mtc=mtc,
            title='Dialogue response evaluation',
            description='Rate a response in a dialogue',
            keywords='evaluation, dialogue',
            reward=0.10,
            )

    if args.review:
        results = task.get_reviewable_results()
        print '{} results available'.format(len(results))
        write_json(results, 'eval_results.json')
    else:
        hit_questions = []
        for questions in data.iter(num_unique_context=50, num_evals_per_instance=10, num_instances_per_batch=10):
            q = task.build_question_form(questions)
            hit_questions.append(q)
        decision = raw_input('About to create {} HITs. Continue? [Y/N]'.format(len(hit_questions)))
        if decision == 'Y':
            for q in hit_questions:
                hit_id = task.create_hit(q)
            print "Your HIT has been created. You can see it at this link:"
            print "https://workersandbox.mturk.com/mturk/preview?groupId={}".format(task.hit_type_id)
        else:
            print 'Abort'
