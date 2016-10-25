import argparse
from collections import defaultdict
import json
import sqlite3
from datetime import datetime
import os
import shutil
import warnings
import atexit

from src.basic.scenario_db import add_scenario_arguments, ScenarioDB
from src.basic.schema import Schema
from src.basic.event import Event
from src.basic.dataset import Example
from src.basic.kb import KB
from src.basic.util import read_json
from src.web import create_app
from src.basic.systems.simple_system import SimpleSystem
from src.basic.systems.heuristic_system import HeuristicSystem
from src.basic.systems.neural_system import NeuralSystem
from src.basic.systems.human_system import HumanSystem
from main import backend
from gevent.wsgi import WSGIServer

__author__ = 'anushabala'

DB_FILE_NAME = 'chat_state.db'
LOG_FILE_NAME = 'log.out'
TRANSCRIPTS_DIR = 'transcripts'


def add_website_arguments(parser):
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host IP address to run app on. Defaults to localhost.')
    parser.add_argument('--config', type=str, default='app_params.json',
                        help='Path to JSON file containing configurations for website')
    parser.add_argument('--output', type=str,
                        default="web_output/{}".format(datetime.now().strftime("%Y-%m-%d")),
                        help='Name of directory for storing website output (debug and error logs, chats, '
                             'and database). Defaults to a web_output/current_data, with the current date formatted as '
                             '%%Y-%%m-%%d. '
                             'If the provided directory exists, all data in it is overwritten.')
    parser.add_argument('--domain', type=str,
                        choices=['MutualFriends', 'Matchmaking'])


def init_database(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE active_user (name text unique, status string, status_timestamp integer,
        connected_status integer, connected_timestamp integer, message text, room_id integer, partner_type text,
        partner_id text, scenario_id text, agent_index integer, selected_index integer, chat_id text)''')
    c.execute('''CREATE TABLE mturk_task (name text, mturk_code text, chat_id text)''')
    c.execute(
        '''CREATE TABLE survey (name text, chat_id text, partner_type text, how_mechanical integer,
        how_effective integer)''')
    c.execute('''CREATE TABLE event (chat_id text, action text, agent integer, time text, data text)''')
    c.execute('''CREATE TABLE chat (chat_id text, scenario_id text, outcome text)''')

    conn.commit()
    conn.close()


def add_systems(config_dict, schema, lexicon):
    """
    Params:
    config_dict: A dictionary that maps the bot name to a dictionary containing configs for the bot. The
        dictionary should contain the bot type (key 'type') and. for bots that use an underlying model for generation,
        the path to the directory containing the parameters, vocab, etc. for the model.
    Returns:
    agents: A dict mapping from the bot name to the System object for that bot.
    pairing_probabilities: A dict mapping from the bot name to the probability that a user is paired with that
        bot. Also includes the pairing probability for humans (backend.Partner.Human)
    """

    systems = {backend.Partner.Human: HumanSystem()}

    for (sys_name, info) in config_dict.iteritems():
        if info["active"]:
            type = info["type"]
            if type == SimpleSystem.name():
                model = SimpleSystem()
            elif type == HeuristicSystem.name():
                model = HeuristicSystem()
            elif type == NeuralSystem.name():
                path = info["path"]
                model = NeuralSystem(schema, lexicon, path)
            else:
                warnings.warn(
                    'Unrecognized model type in {} for configuration '
                    '{}. Ignoring configuration.'.format(info, sys_name))
                continue
            systems[sys_name] = model

    prob = 1.0/len(systems.keys())
    pairing_probabilities = {system_name: prob for system_name in systems.keys()}

    return systems, pairing_probabilities


def cleanup(flask_app):
    db_path = flask_app.config['user_params']['db']['location']
    transcript_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'transcripts.json')
    log_events_to_json(app.config['scenario_db'], db_path, transcript_path)


def log_events_to_json(scenario_db, db_path, json_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # c.execute('''CREATE TABLE event (chat_id text, action text, agent integer, time text, data text)''')
    cursor.execute('SELECT DISTINCT chat_id FROM event')
    ids = cursor.fetchall()

    examples = []
    for chat_id in ids:
        # chat_id is a tuple   (id,)
        cursor.execute('SELECT agent, action, time, data FROM event WHERE chat_id=? ORDER BY time ASC', chat_id)
        logged_events = cursor.fetchall()
        cursor.execute('SELECT scenario_id, outcome FROM chat WHERE chat_id=?', chat_id)
        (uuid, outcome) = cursor.fetchone()
        outcome = json.loads(outcome)
        chat_events = []
        for (agent, action, time, data) in logged_events:
            if action == 'join' or action == 'leave':
                continue
            if action == 'select':
                data = KB.string_to_ordered_item(data)
            event = Event(agent, time, action, data)
            chat_events.append(event)
        ex = Example(scenario_db.get(uuid), uuid, chat_events, outcome)
        examples.append(ex)

    outfile = open(json_path, 'w')
    json.dump([ex.to_dict() for ex in examples], outfile)
    outfile.close()
    conn.close()


def init(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    db_file = os.path.join(output_dir, DB_FILE_NAME)
    init_database(db_file)

    log_file = os.path.join(output_dir, LOG_FILE_NAME)

    transcripts_dir = os.path.join(output_dir, TRANSCRIPTS_DIR)
    if os.path.exists(transcripts_dir):
        shutil.rmtree(transcripts_dir)

    os.makedirs(transcripts_dir)

    return db_file, log_file, transcripts_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_website_arguments(parser)
    add_scenario_arguments(parser)
    args = parser.parse_args()

    params_file = args.config
    with open(params_file) as fin:
        params = json.load(fin)

    db_file, log_file, transcripts_dir = init(args.output)
    params['db'] = {}
    params['db']['location'] = db_file
    params['logging'] = {}
    params['logging']['app_log'] = log_file
    params['logging']['chat_dir'] = transcripts_dir

    if 'task_title' not in params.keys():
        raise ValueError("Title of task should be specified in config file with the key 'task_title'")

    instructions = None
    if 'instructions' in params.keys():
        instructions_file = open(params['instructions'], 'r')
        instructions = "".join(instructions_file.readlines())
        instructions_file.close()
    else:
        raise ValueError("Location of file containing instructions for task should be specified in config with the key "
                         "'instructions")

    templates_dir = None
    if 'templates_dir' in params.keys():
        templates_dir = params['templates_dir']
    else:
        raise ValueError("Location of HTML templates should be specified in config with the key templates_dir")
    if not os.path.exists(templates_dir):
            raise ValueError("Specified HTML template location doesn't exist: %s" % templates_dir)

    app = create_app(debug=True, templates_dir=templates_dir)

    schema_path = args.schema_path

    if not os.path.exists(schema_path):
        raise ValueError("No schema file found at %s" % schema_path)

    schema = Schema(schema_path, domain=args.domain)
    # todo in the future would we want individual models to have different lexicons?
    lexicon = None
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    app.config['scenario_db'] = scenario_db

    if 'models' not in params.keys():
        params['models'] = {}

    systems, pairing_probabilities = add_systems(params['models'], schema, lexicon)

    app.config['systems'] = systems
    app.config['sessions'] = defaultdict(None)
    app.config['pairing_probabilities'] = pairing_probabilities
    app.config['schema'] = schema
    app.config['lexicon'] = lexicon
    app.config['user_params'] = params
    app.config['sessions'] = defaultdict(None)
    app.config['controller_map'] = defaultdict(None)
    app.config['instructions'] = instructions
    app.config['task_title'] = params['task_title']
    atexit.register(cleanup, flask_app=app)

    server = WSGIServer(('', args.port), app)
    server.serve_forever()
