import random
import string
from argparse import ArgumentParser

from cocoa.core.util import read_json, write_json
from cocoa.turk.utils import get_mturk_connection
from cocoa.turk.task import HTMLEvalTask, HTMLCompareEvalTask, add_turk_task_arguments

from turk.eval_data import EvalData, add_eval_data_arguments


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--review', action='store_true', help='Review completed hits')
    parser.add_argument('--check-workers', action='store_true', help='Detect spammers')
    parser.add_argument('--remove-evaluated', nargs='*', default=[], help='Path to JSON file that saves old result')
    parser.add_argument('--qids', help='Path to JSON file that contains question/context ids to be evaluated')
    parser.add_argument('--debug', action='store_true', help='If ture, run in sandbox')
    parser.add_argument('--aws-config', required=True, help='AWS credential')
    parser.add_argument('--num-eval', type=int, default=20, help='Number of unique context to evaluate')
    parser.add_argument('--systems', nargs='*', default=None, help='Systems to evaluate')
    parser.add_argument('--num-test-questions', type=int, default=0, help='Number of questions of which we know the answers')
    parser.add_argument('--num-overlap-questions', type=int, default=10, help='Number of questions repeated in each HIT for estimating agreement')
    parser.add_argument('--compare', action='store_true', help='Rate by comparing two responses')
    parser.add_argument('--title', default='Dialogue response evaluation', help='Title of the HIT')
    add_turk_task_arguments(parser)
    add_eval_data_arguments(parser)
    args = parser.parse_args()

    random.seed(1)

    config = read_json(args.aws_config)
    mtc = get_mturk_connection(config, debug=args.debug)

    if args.debug:
        lifetime = 0.1
        args.hit_db = 'test.json'
    else:
        lifetime = 30
    if args.compare:
        Task = HTMLCompareEvalTask
    else:
        Task = HTMLEvalTask
    task = Task(mtc=mtc,
            title=args.title,
            description='Rate a response in a dialogue',
            keywords='evaluation, dialogue',
            reward=args.reward_per_hit,
            question_title='Rate responses given the conversational context',
            overall_template_path=args.overall_template,
            question_template_path=args.question_template,
            instructions_path=args.instructions,
            script_path=args.script,
            max_assignments=args.num_assignments_per_hit,
            db_path=args.hit_db,
            lifetime=lifetime,
            qualifications='default',
            )

    if args.check_workers:
        task.check_workers()
    elif args.review:
        task.get_reviewable_results()
    else:
        if args.eval_data:
            print 'Load saved processed evaluation data'
            data = EvalData.from_json(args.eval_data)
        else:
            assert len(args.system_outputs) % 2 == 0
            x = args.system_outputs
            eval_data = [(x[i], x[i+1]) for i in range(0, len(x), 2)]
            print 'Read eval data from generate files'
            data = EvalData.from_file(eval_data, args.num_context)
            data.dump(args.eval_data_output)

        evaluated_qids = set()
        for db in args.remove_evaluated:
            qids = task.get_evaluated_qids(db)
            evaluated_qids.update(qids)
        if args.compare:
            pairs = ['ir', 'generative']
        else:
            pairs = None
        if args.qids:
            qids = read_json(args.qids)
        else:
            qids = None
        questions, selected_ids = data.sample_examples(args.num_eval, evaluated=evaluated_qids, systems=args.systems, pairs=pairs, qids=qids)
        if args.num_overlap_questions > 0:
            evaluated_qids.update(selected_ids)
            test_questions, _ = data.sample_examples(args.num_overlap_questions, evaluated=evaluated_qids, systems=args.systems, pairs=pairs)
        #references, selected_ids = data.sample_examples(args.num_eval, evaluated=evaluated_qids, systems=['reference'])
        #random.shuffle(questions)

        #task.launch_hits(*task.create_questions([questions]))
        #import sys; sys.exit()

        num_questions_per_hit = args.num_questions_per_hit - args.num_test_questions
        if (not args.compare and (args.num_eval * len(args.systems)) % num_questions_per_hit != 0) or \
            (args.compare and args.num_eval % num_questions_per_hit != 0):
            print 'Evaluting {} contexts with {} questions per HIT - not divisible. Possible to get fewer questions in a HIT'.format(args.num_eval, args.num_questions_per_hit)
            raise ValueError

        batch_questions = []
        j = 0
        for i in xrange(0, len(questions), num_questions_per_hit):
            batch = questions[i: i+num_questions_per_hit]
            batch.extend(test_questions)
            #batch.extend(references[j: j+args.num_test_questions])
            #j += args.num_test_questions
            batch_questions.append(batch)
        task.launch_hits(*task.create_questions(batch_questions))
