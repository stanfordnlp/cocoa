import sys
import os

from boto.mturk.question import QuestionContent, Question, QuestionForm, Overview, AnswerSpecification, SelectionAnswer, FormattedContent, FreeTextAnswer, HTMLQuestion

from cocoa.core.util import read_json, write_json
from utils import default_qualifications, xml_safe

def add_turk_task_arguments(parser):
    parser.add_argument('--overall-template', help='Path to HTML templates')
    parser.add_argument('--question-template', help='Path to HTML templates')
    parser.add_argument('--instructions', help='Path to instructions')
    parser.add_argument('--num-assignments-per-hit', type=int, default=3, help='How many lables do we want on each example')
    parser.add_argument('--num-questions-per-hit', type=int, default=10, help='How many questions in a hit')
    parser.add_argument('--reward-per-hit', type=float, default=0.10, help='Payment per hit in dollar')
    parser.add_argument('--hit-db', default='hits.json', help='Path to JSON file that saves hits info')

class Task(object):
    def __init__(self, mtc=None, title=None, description=None, keywords=None, reward=None, max_assignments=1, qualifications=None, duration=60, approval_delay=3, lifetime=30, db_path='hits.json'):
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
        if qualifications is None:
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

    def launch_hits(self, questions):
        decision = raw_input('About to create {} HITs that costs ${}. Continue? [Y/N]'.format(len(questions), self.reward * len(questions) * self.max_assignments))
        if decision == 'Y':
            for q in questions:
                hit_id = self.create_hit(q)
                self.db[hit_id] = {}
            self.dump_db()
            print "Your HIT has been created. You can see it at this link:"
            print "https://workersandbox.mturk.com/mturk/preview?groupId={}".format(self.hit_type_id)
        else:
            print 'Abort'

    def get_reviewable_results(self):
        results = []
        for hit_id, hit_info in self.db.iteritems():
            assignments = self.mtc.get_assignments(hit_id)
            if assignments:
                for assignment in assignments:
                    answers = []
                    for answer in assignment.answers[0]:
                        answers.append({'qid': answer.qid, 'answer': answer.fields[0]})
                    hit_info[assignment.AssignmentId] = {
                            'worker_id': assignment.WorkerId,
                            'answers': answers,
                            }
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
            display_context.append(self.utterance_formatter(linebreak=True, color=color).format(agent=agent, text=utterance))
        return ''.join(display_context)


class HTMLEvalTask(EvalTask):
    def __init__(self, question_template_path=None, overall_template_path=None, instructions_path=None, question_title='Dialogue evaluation', **kwargs):
        super(HTMLEvalTask, self).__init__(**kwargs)
        with open(overall_template_path, 'r') as fin:
            self.overall_template = fin.read().strip()
        with open(question_template_path, 'r') as fin:
            self.question_template = fin.read().strip()
        with open(instructions_path, 'r') as fin:
            self.instructions = fin.read().strip()
        self.title = question_title

    def create_questions(self, questions):
        hit_questions = []
        for question_group in questions:
            html_questions = self.create_question_group(question_group)
            html_hit = self.overall_template.format(
                    title=self.title,
                    instructions=self.instructions,
                    questions=html_questions,
                    )
            html_hit = HTMLQuestion(html_hit, 600)
            hit_questions.append(html_hit)
        return hit_questions

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


################ DEPRECATED ################

class QuestionFormEvalTask(EvalTask):
    ratings = [
            ('Very unlikely', -2),
            ('Not likely', -1),
            ('Neutral', 0),
            ('Likely', 1),
            ('Very likely', 2),
            ]

    def build_question(self, qid, context, response):
        qc = QuestionContent()

        utterance_formatter = '{agent}: {text}'
        #qc.append_field('Text', 'Dialogue context:')
        display_context = []
        for agent, utterance in context:
            display_context.append(self.utterance_formatter(linebreak=True).format(agent=agent, text=utterance))
        qc.append(FormattedContent(''.join(display_context)))

        #qc.append_field('Text', 'Response:')
        agent, utterance = response
        qc.append(FormattedContent(self.utterance_formatter().format(agent=agent, text=utterance)))

        qc.append_field('Text', 'Please rate how likely the response (in bold) continues from the conversation above.')

        ans = SelectionAnswer(min=-2, max=2, style='radiobutton',
                selections=self.ratings,
                type='text', other=False)

        q = Question(identifier=qid,
                content=qc,
                answer_spec=AnswerSpecification(ans),
                is_required=True)

        return q

    def build_comment(self):
        qc = QuestionContent()
        qc.append_field('Text', 'Comments:')
        ans = FreeTextAnswer()
        q = Question(identifier='comment',
              content=qc,
              answer_spec=AnswerSpecification(ans),
              is_required=False)
        return q

    def build_question_form(self, questions):
        question_form = QuestionForm()
        question_form.append(self.build_overview())
        for qid, context, response in questions:
            question_form.append(self.build_question(qid, context, response))
            question_form.append(self.build_comment())
        return question_form

    def build_overview(self,):
        overview = Overview()
        overview.append_field('Title', 'Rate a response in a dialogue')
        overview.append(FormattedContent('''
                <p>You are given an excerpt of a conversation (the context)
                happened between a buyer and a seller negotiating the price of an
                item for sale, and a response following the context.
                Your task is to decide how likely the response (<b>in bold</b>)
                and the context actually come from the same dialogue, i.e. together
                they form a natual, coherent dialogue.</p>
                <p>Mentions of prices are replaced by the symbol <b>PRICE</b>;
                you can assume the actual amount is reasonable. </p>
                <p>Symbols such as OFFER, ACCEPT, REJECT denote actions.</p>
                <p><b>NOTE:</b> We use verification answers to identify abusers.
                Randomly chosen answers will <b>get rejected</b>.</p>
        '''))
        return overview
