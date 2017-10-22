import sys
import os
from collections import defaultdict
from itertools import izip

from boto.mturk.question import QuestionContent, Question, QuestionForm, Overview, AnswerSpecification, SelectionAnswer, FormattedContent, FreeTextAnswer, HTMLQuestion
from boto.mturk.connection import MTurkRequestError

from cocoa.core.util import read_json, write_json
from utils import default_qualifications, xml_safe

def add_turk_task_arguments(parser):
    parser.add_argument('--overall-template', help='Path to HTML templates')
    parser.add_argument('--question-template', help='Path to HTML templates')
    parser.add_argument('--instructions', help='Path to instructions')
    parser.add_argument('--script', help='Path to .js script')
    parser.add_argument('--num-assignments-per-hit', type=int, default=3, help='How many lables do we want on each example')
    parser.add_argument('--num-questions-per-hit', type=int, default=10, help='How many questions in a hit')
    parser.add_argument('--reward-per-hit', type=float, default=0.10, help='Payment per hit in dollar')
    parser.add_argument('--hit-db', default='hits.json', help='Path to JSON file that saves hits info')

class Task(object):
    def __init__(self, mtc=None, title=None, description=None, keywords=None, reward=None, max_assignments=1, qualifications='default', duration=60, approval_delay=3, lifetime=30, db_path='hits.json'):
        """Initialize a Turk task with basic properties.

        Args:
            mtc (MTurkConnection)
            title (str)
            description (str)
            keywords (str)
            reward (float): in dollars, e.g. 0.50.
            max_assignments (int): how many times can the HIT be completed.
            qualifications (Qualifications)
            duration (int): in minutes.
            approval_delay (int): in days.
            lifetime (int): in days.
            db_path (str): file for saving hit_ids and results.

        """
        self.title = title
        self.description = description
        self.keywords = keywords
        self.reward = reward
        self.max_assignments = max_assignments
        if qualifications == 'default':
            qualifications = default_qualifications()
        self.qualifications = qualifications
        self.duration = duration * 60
        self.approval_delay = approval_delay * 24 * 60 * 60
        self.lifetime = lifetime * 24 * 60 * 60

        self.mtc = mtc
        hit_type_ids = self.mtc.register_hit_type(self.title, self.description, self.reward, self.duration, self.keywords, self.approval_delay, self.qualifications)
        self.hit_type_id = hit_type_ids[0].HITTypeId

        self.db_path = db_path
        self.load_db()

    def load_db(self):
        db_path = self.db_path
        if os.path.exists(db_path):
            self.db = read_json(db_path)
        else:
            self.db = {}

    def dump_db(self):
        write_json(self.db, self.db_path)
        print 'HIT results dumped to {}'.format(self.db_path)

    def create_hit(self, question):
        hit = self.mtc.create_hit(hit_type=self.hit_type_id, question=question, lifetime=self.lifetime, max_assignments=self.max_assignments)
        return hit[0].HITId

    def launch_hits(self, questions, question_data):
        """Launch HITs to AMT.

        Args:
            questions (list[HTMLQuestion]): (see boto)
            question_data (list[JSON dict]): to be written to db

        """
        assert len(questions) == len(question_data)
        decision = raw_input('About to create {} HITs that costs ${}. Continue? [Y/N]'.format(len(questions), self.reward * len(questions) * self.max_assignments))
        if decision == 'Y':
            for q, q_data in izip(questions, question_data):
                hit_id = self.create_hit(q)
                self.db[hit_id] = {'data': q_data}
            self.dump_db()
            print "Your HIT has been created. You can see it at this link:"
            print "https://workersandbox.mturk.com/mturk/preview?groupId={}".format(self.hit_type_id)
        else:
            print 'Abort'

    def check_workers(self):
        """Print workers' answers to check spammers.
        """
        worker_answers = defaultdict(lambda : defaultdict(int))
        for hit_id, hit_info in self.db.iteritems():
            for assignment_id, result in hit_info.iteritems():
                worker_id = result['worker_id']
                answers = result['answers']
                for answer in answers:
                    if answer['qid'] == 'comment':
                        continue
                    worker_answers[worker_id][int(answer['answer'])] += 1
        for worker, answers in worker_answers.iteritems():
            print worker, [answers[score] for score in xrange(-2, 3)]

    @classmethod
    def get_evaluated_qids(cls, db_path):
        db = read_json(db_path)
        qids = set()
        for hit_id, hit_info in db.iteritems():
            for assignment_id, result in hit_info.iteritems():
                answers = result['answers']
                for answer in answers:
                    if answer['qid'] == 'comment':
                        continue
                    qids.add(answer['qid'])
        return qids

    def get_reviewable_results(self):
        """Get HIT results.

        Results are written in `self.db` in the following JSON structure:
            |-hit_id
              |-assignment_id
                |-"worker_id"
                |-"answers"
                  |-[i]
                    |-"answer"
                    |-"qid"

        """
        results = []
        num_reviewable_assignments = 0
        for hit_id, hit_info in self.db.iteritems():
            try:
                assignments = self.mtc.get_assignments(hit_id)
                print hit_id, len(assignments)
            except MTurkRequestError:
                continue
            if assignments:
                for assignment in assignments:
                    answers = []
                    for answer in assignment.answers[0]:
                        answers.append({'qid': answer.qid, 'answer': answer.fields[0]})
                    hit_info[assignment.AssignmentId] = {
                            'worker_id': assignment.WorkerId,
                            'answers': answers,
                            }
                    num_reviewable_assignments += 1
        print '{} assignments ready'.format(num_reviewable_assignments)
        self.dump_db()

class EvalTask(Task):
    def utterance_formatter(self, linebreak=False, color=None):
        template = '{agent}: {text}'
        if color:
            template = '<span style=color:{color}>{temp}</span>'.format(color=color, temp=template)
        if linebreak:
            template = '{}<br>'.format(template)
        return template

    def format_context(self, context):
        utterance_formatter = '{agent}: {text}'
        display_context = []
        for i, (agent, utterance) in enumerate(context):
            color = 'red' if i % 2 == 0 else 'green'
            display_context.append(self.utterance_formatter(linebreak=True, color=color).format(agent=agent, text=utterance.encode('utf-8')))
        return ''.join(display_context)


class HTMLEvalTask(EvalTask):
    def __init__(self, question_template_path=None, overall_template_path=None, instructions_path=None, script_path=None, question_title='Dialogue evaluation', **kwargs):
        super(HTMLEvalTask, self).__init__(**kwargs)
        with open(overall_template_path, 'r') as fin:
            self.overall_template = fin.read().strip()
        with open(question_template_path, 'r') as fin:
            self.question_template = fin.read().strip()
        with open(instructions_path, 'r') as fin:
            self.instructions = fin.read().strip()
        if script_path is not None:
            with open(script_path, 'r') as fin:
                self.script = fin.read().strip()
        else:
            self.script = ''
        self.title = question_title

    def create_instructions(self, batch_size):
        return self.instructions.format(batch_size=batch_size)

    def create_questions(self, questions):
        hit_questions = []
        question_data = []
        for question_group in questions:
            html_questions = self.create_question_group(question_group)
            html_hit = self.overall_template.format(
                    title=self.title,
                    instructions=self.create_instructions(len(question_group)),
                    script=self.script,
                    questions=html_questions,
                    )
            #with open('/afs/cs.stanford.edu/u/hehe/www/question.html', 'w') as fout:
            #    fout.write(html_hit)
            #    import sys; sys.exit()
            html_hit = HTMLQuestion(html_hit, 600)
            hit_questions.append(html_hit)
            question_data.append(self.json_question_data(question_group))
        return hit_questions, question_data

    def json_question_data(self, questions):
        """JSON dict of questions in a HIT.
        """
        data = {qid: {'context': context, 'response': response} for qid, context, response in questions}
        return data

    def create_question_group(self, questions):
        html_questions = []
        for qid, context, response in questions:
            html_context = self.format_context(context)
            agent, utterance = response
            html_response = self.utterance_formatter().format(agent=agent, text=utterance.encode('utf-8'))
            html_question = self.question_template.format(
                qid=qid,
                context=html_context,
                response=html_response,
                )
            html_questions.append(html_question)
        return '\n'.join(html_questions)

class HTMLCompareEvalTask(HTMLEvalTask):
    def create_question_group(self, questions):
        html_questions = []
        for qid, context, responses in questions:
            html_context = self.format_context(context)
            html_responses = []
            for response in responses:
                agent, utterance = response
                html_response = self.utterance_formatter().format(agent=agent, text=utterance.encode('utf-8'))
                html_responses.append(html_response)
            html_question = self.question_template.format(
                qid=qid,
                context=html_context,
                response0=html_responses[0],
                response1=html_responses[1],
                )
            html_questions.append(html_question)
        return '\n'.join(html_questions)

