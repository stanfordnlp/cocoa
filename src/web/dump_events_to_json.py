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


def log_events_to_json(scenario_db, db_path, json_path, uids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # c.execute('''CREATE TABLE event (chat_id text, action text, agent integer, time text, data text)''')
    if args.uid is None:
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
        # chat_id is a tuple   (id,)
        cursor.execute('SELECT agent, action, time, data, start_time FROM event WHERE chat_id=? ORDER BY time ASC', chat_id)
        logged_events = cursor.fetchall()
        cursor.execute('SELECT scenario_id, outcome FROM chat WHERE chat_id=?', chat_id)
        (uuid, outcome) = cursor.fetchone()
        try:
            outcome = json.loads(outcome)
        except ValueError:
            continue
        chat_events = []
        for (agent, action, time, data, start_time) in logged_events:
            if action == 'join' or action == 'leave':
                continue
            if action == 'select':
                data = KB.string_to_item(data)
            event = Event(agent, time, action, data, start_time)
            chat_events.append(event)
        ex = Example(scenario_db.get(uuid), uuid, chat_events, outcome, chat_id[0])
        examples.append(ex)

    outfile = open(json_path, 'w')
    json.dump([ex.to_dict() for ex in examples], outfile)
    outfile.close()
    conn.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    parser.add_argument('--db', type=str, required=True, help='Path to database file containing logged events')
    parser.add_argument('--domain', type=str,
                        choices=['MutualFriends', 'Matchmaking'])
    parser.add_argument('--output', type=str, required=True, help='File to write JSON examples to.')
    parser.add_argument('--uid', type=str, nargs='*', help='Only print chats from these uids')
    args = parser.parse_args()
    schema = Schema(args.schema_path, args.domain)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))

    log_events_to_json(scenario_db, args.db, args.output, args.uid)
