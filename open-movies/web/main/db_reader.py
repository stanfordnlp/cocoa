import math
import json

from cocoa.web.main.db_reader import DatabaseReader as BaseDatabaseReader
from cocoa.core.util import write_json

class DatabaseReader(BaseDatabaseReader):
    @classmethod
    def get_chat_outcome(cls, cursor, chat_id):
        outcome = super(DatabaseReader, cls).get_chat_outcome(cursor, chat_id)
        return outcome

    @classmethod
    def dump_surveys(cls, cursor, json_path):
        questions = ['humanlike', 'interesting', 'comments']

        cursor.execute('''SELECT * FROM survey''')
        logged_surveys = cursor.fetchall()
        survey_data = {}
        agent_types = {}

        for survey in logged_surveys:
            # todo this is pretty lazy - support variable # of questions per task eventually..
            (userid, cid, q1, q2, comments) = survey
            responses = dict(zip(questions, [q1, q2, comments]))
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
