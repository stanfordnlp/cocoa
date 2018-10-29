import math
import json

from cocoa.web.main.db_reader import DatabaseReader as BaseDatabaseReader
from cocoa.core.util import write_json

class DatabaseReader(BaseDatabaseReader):
    @classmethod
    def get_chat_outcome(cls, cursor, chat_id):
        outcome = super(DatabaseReader, cls).get_chat_outcome(cursor, chat_id)
        try:
            if math.isnan(outcome['book']):
                outcome['book'] = None
            if math.isnan(outcome['hat']):
                outcome['hat'] = None
            if math.isnan(outcome['ball']):
                outcome['ball'] = None
        except (ValueError, TypeError, KeyError) as e:
            pass
        return outcome

    @classmethod
    def get_chat_example(cls, cursor, chat_id, scenario_db):
        ex = super(DatabaseReader, cls).get_chat_example(cursor, chat_id, scenario_db)
        if not ex is None:
            cursor.execute('SELECT config FROM bot where chat_id=?', (chat_id,))
            result = cursor.fetchone()
            if result:
                ex.agents_info = {'config': result[0]}
        return ex

    @classmethod
    def process_event_data(cls, action, data):
        if action == 'select':
            data = json.loads(data)
            try:
                if math.isnan(data['book']):
                    data['book'] = None
                if math.isnan(data['hat']):
                    data['hat'] = None
                if math.isnan(data['ball']):
                    data['ball'] = None
            except (ValueError, TypeError) as e:
                pass
        return data

    @classmethod
    def dump_surveys(cls, cursor, json_path):
        questions = ['negotiator', 'comments']

        cursor.execute('''SELECT * FROM survey''')
        logged_surveys = cursor.fetchall()
        survey_data = {}
        agent_types = {}

        for survey in logged_surveys:
            # todo this is pretty lazy - support variable # of questions per task eventually..
            (userid, cid, q1, comments) = survey
            responses = dict(zip(questions, [q1, comments]))
            cursor.execute('''SELECT agent_types, agent_ids FROM chat WHERE chat_id=?''', (cid,))
            chat_result = cursor.fetchone()
            agents = json.loads(chat_result[0])
            agent_ids = json.loads(chat_result[1])
            agent_types[cid] = agents
            if cid not in survey_data.keys():
                survey_data[cid] = {0: {}, 1: {}}
            partner_idx = 0 if agent_ids['1'] == userid else 1
            survey_data[cid][partner_idx] = responses

        write_json([agent_types, survey_data], json_path)
