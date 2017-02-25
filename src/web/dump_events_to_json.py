from src.basic.systems.human_system import HumanSystem

__author__ = 'anushabala'
import sqlite3
import json
from src.basic.event import Event
from src.basic.dataset import Example
from src.basic.kb import KB
from argparse import ArgumentParser
from src.basic.scenario_db import add_scenario_arguments, ScenarioDB
from src.basic.schema import Schema
from src.basic.util import read_json
from datetime import datetime

date_fmt = '%Y-%m-%d %H-%M-%S'


def convert_events_to_json(chat_id, cursor, scenario_db):
    try:
        cursor.execute('SELECT agent, action, time, data, start_time FROM event WHERE chat_id=? ORDER BY time ASC', (chat_id,))
        logged_events = cursor.fetchall()
    except sqlite3.OperationalError:
        cursor.execute('SELECT agent, action, time, data FROM event WHERE chat_id=? ORDER BY time ASC', (chat_id,))
        logged_events = cursor.fetchall()
        events = []
        for i, (agent, action, time, data) in enumerate(logged_events):
            events.append((agent, action, time, data, time))
        logged_events = events
    cursor.execute('SELECT scenario_id, outcome FROM chat WHERE chat_id=?', (chat_id,))
    (uuid, outcome) = cursor.fetchone()
    try:
        outcome = json.loads(outcome)
    except ValueError:
        outcome = {'reward': 0}

    try:
        cursor.execute('SELECT agent_types FROM chat WHERE chat_id=?', (chat_id,))
        agent_types = cursor.fetchone()[0]
        agent_types = json.loads(agent_types)
    except sqlite3.OperationalError:
        agent_types = {0: HumanSystem.name(), 1: HumanSystem.name()}

    chat_events = []
    for (agent, action, time, data, start_time) in logged_events:
        if action == 'join' or action == 'leave':
            continue
        if action == 'select':
            data = KB.string_to_item(data)

        time = convert_time_format(time)
        start_time = convert_time_format(start_time)
        event = Event(agent, time, action, data, start_time)
        chat_events.append(event)
    return Example(scenario_db.get(uuid), uuid, chat_events, outcome, chat_id, agent_types)


def log_transcripts_to_json(scenario_db, db_path, json_path, uids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # c.execute('''CREATE TABLE event (chat_id text, action text, agent integer, time text, data text)''')
    if uids is None:
        cursor.execute('SELECT DISTINCT chat_id FROM event')
        ids = cursor.fetchall()
    else:
        ids = []
        uids = [(x,) for x in uids]
        for uid in uids:
            cursor.execute('SELECT chat_id FROM mturk_task WHERE name=?', uid)
            ids_ = cursor.fetchall()
            ids.extend(ids_)

    examples = []
    for chat_id in ids:
        ex = convert_events_to_json(chat_id[0], cursor, scenario_db)
        examples.append(ex)

    outfile = open(json_path, 'w')
    json.dump([ex.to_dict() for ex in examples], outfile)
    outfile.close()
    conn.close()


def log_surveys_to_json(db_path, surveys_file):
    questions = ['fluent', 'correct', 'cooperative', 'humanlike', 'comments']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM survey''')
    logged_surveys = cursor.fetchall()
    survey_data = {}
    agent_types = {}

    for survey in logged_surveys:
        # print survey
        (userid, cid, _, fluent, correct, cooperative, humanlike, comments) = survey
        responses = dict(zip(questions, [fluent, correct, cooperative, humanlike, comments]))
        cursor.execute('''SELECT agent_types, agent_ids FROM chat WHERE chat_id=?''', (cid,))
        chat_result = cursor.fetchone()
        agents = json.loads(chat_result[0])
        agent_ids = json.loads(chat_result[1])
        agent_types[cid] = agents
        if cid not in survey_data.keys():
            survey_data[cid] = {0: {}, 1: {}}
        partner_idx = 0 if agent_ids['1'] == userid else 1
        survey_data[cid][partner_idx] = responses

    json.dump([agent_types, survey_data], open(surveys_file, 'w'))



def convert_time_format(time):
    if time is None:
        return time
    try:
        dt = datetime.strptime(time, date_fmt)
        s = str((dt - datetime.fromtimestamp(0)).total_seconds())
        return s
    except (ValueError, TypeError):
        try:
            dt = datetime.fromtimestamp(float(time)) # make sure that time is a UNIX timestamp
            return time
        except (ValueError, TypeError):
            print 'Unrecognized time format: %s' % time

    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    parser.add_argument('--db', type=str, required=True, help='Path to database file containing logged events')
    parser.add_argument('--domain', type=str,
                        choices=['MutualFriends', 'Matchmaking'])
    parser.add_argument('--output', type=str, required=True, help='File to write JSON examples to.')
    parser.add_argument('--uid', type=str, nargs='*', help='Only print chats from these uids')
    parser.add_argument('--surveys', type=str, help='If provided, writes a file containing results from user surveys.')
    args = parser.parse_args()
    schema = Schema(args.schema_path, args.domain)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))

    log_transcripts_to_json(scenario_db, args.db, args.output, args.uid)
    if args.surveys:
        log_surveys_to_json(args.db, args.surveys)
