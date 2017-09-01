import sys

from boto.mturk.question import QuestionContent, Question, QuestionForm, Overview, AnswerSpecification, SelectionAnswer, FormattedContent, FreeTextAnswer

from cocoa.core.util import read_json, write_json
from utils import default_qualifications, xml_safe

class Task(object):
    def __init__(self, mtc=None, title=None, description=None, keywords=None, reward=None, max_assignments=1, qualifications=None, duration=60, approval_delay=3, lifetime=30):
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

    def create_hit(self, question):
        hit = self.mtc.create_hit(hit_type=self.hit_type_id, question=question, lifetime=self.lifetime, max_assignments=self.max_assignments)
        return hit[0].HITId

    # TODO: save hit_ids when creating!
    def get_reviewable_results(self, N=100):
        current_page = 1
        page_size=100
        my_hits = []
        while True:
            search_rs = self.mtc.search_hits(page_size=page_size, page_number=current_page, sort_direction='Descending')
            for hit in search_rs:
                if hit.HITTypeId == self.hit_type_id and hit.HITStatus == 'Reviewable':
                    my_hits.append(hit)
            if len(my_hits) >= N or current_page >= 10:
                break
            else:
                current_page += 1

        results = []
        for hit in my_hits:
            hit_id = hit.HITId
            result = self.mtc.get_assignments(hit_id)
            assignment = result[0]
            worker_id = assignment.WorkerId
            answers = []
            for answer in assignment.answers[0]:
                answers.append({'qid': answer.qid, 'answer': answer.fields[0]})
            results.append({
                'hit_id': hit_id,
                'worker_id': worker_id,
                'answers': answers,
                })
        return results


class EvalTask(Task):
    ratings = [
            ('Very unlikely', -2),
            ('Not likely', -1),
            ('Neutral', 0),
            ('Likely', 1),
            ('Very likely', 2),
            ]

    def utterance_formatter(self, bold=False, linebreak=False):
        template = '{role}: {text}'
        if bold:
            template = '<b>{}</b>'.format(template)
        if linebreak:
            template = '{}<br></br>'.format(template)
        return template

    def build_question(self, qid, context, response):
        qc = QuestionContent()

        utterance_formatter = '{role}: {text}'
        #qc.append_field('Text', 'Dialogue context:')
        display_context = []
        for role, utterance in context:
            display_context.append(self.utterance_formatter(linebreak=True).format(role=role, text=utterance))
        qc.append(FormattedContent(''.join(display_context)))

        #qc.append_field('Text', 'Response:')
        role, utterance = response
        qc.append(FormattedContent(self.utterance_formatter(bold=True).format(role=role, text=utterance)))

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
