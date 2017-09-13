import random
import string
from argparse import ArgumentParser

from cocoa.core.util import read_json, write_json
from cocoa.turk.utils import get_mturk_connection
from cocoa.turk.task import HTMLEvalTask, add_turk_task_arguments

from turk.eval_data import EvalData, add_eval_data_arguments


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--review', action='store_true', help='Review completed hits')
    parser.add_argument('--debug', action='store_true', help='If ture, run in sandbox')
    parser.add_argument('--aws-config', required=True, help='AWS credential')
    parser.add_argument('--num-eval', type=int, default=20, help='Number of unique context to evaluate')
    add_turk_task_arguments(parser)
    add_eval_data_arguments(parser)
    args = parser.parse_args()

    if (args.num_eval % args.num_questions_per_hit) != 0:
        print 'Evaluting {} contexts with {} questions per HIT - not divisible. Possible to get fewer questions in a HIT'.format(args.num_eval, args.num_questions_per_hit)
        raise ValueError

    config = read_json(args.aws_config)
    mtc = get_mturk_connection(config, debug=args.debug)

    task = HTMLEvalTask(mtc=mtc,
            title='Dialogue response evaluation',
            description='Rate a response in a dialogue',
            keywords='evaluation, dialogue',
            reward=args.reward_per_hit,
            question_title='Rate responses given the conversational context',
            overall_template_path=args.overall_template,
            question_template_path=args.question_template,
            instructions_path=args.instructions,
            max_assignments=args.num_assignments_per_hit,
            db_path=args.hit_db,
            )

    if args.review:
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

        questions = data.sample_examples(args.num_eval)
        random.shuffle(questions)
        num_questions_per_hit = args.num_questions_per_hit
        batch_questions = []
        for i in xrange(0, len(questions), num_questions_per_hit):
            batch_questions.append(questions[i: i+num_questions_per_hit])
        task.launch_hits(task.create_questions(batch_questions))
