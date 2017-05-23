import argparse
from collections import defaultdict
import json
import sqlite3
from datetime import datetime
import os
import shutil
import warnings
import atexit
from signal import signal, SIGTERM
import sys

from src.basic.scenario_db import add_scenario_arguments, ScenarioDB
from src.basic.schema import Schema
from src.web.dump_events_to_json import log_transcripts_to_json
from src.basic.util import read_json
from src.web import create_app
from src.basic.systems.simple_system import SimpleSystem
from src.basic.systems.neural_system import NeuralSystem
from src.basic.systems.human_system import HumanSystem
from gevent.wsgi import WSGIServer
from src.basic.lexicon import Lexicon, add_lexicon_arguments
from src.basic.inverse_lexicon import InverseLexicon

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
                             'and database). Defaults to a web_output/current_date, with the current date formatted as '
                             '%%Y-%%m-%%d. '
                             'If the provided directory exists, all data in it is overwritten.')
    parser.add_argument('--domain', type=str,
                        choices=['MutualFriends', 'Matchmaking'])


def init_database(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE active_user (name text unique, status string, status_timestamp integer,
        connected_status integer, connected_timestamp integer, message text, partner_type text,
        partner_id text, scenario_id text, agent_index integer, selected_index integer, chat_id text)'''
    )
    c.execute('''CREATE TABLE mturk_task (name text, mturk_code text, chat_id text)''')
    c.execute(
        '''CREATE TABLE survey (name text, chat_id text, partner_type text, fluent integer,
        correct integer, cooperative integer, human_like integer, comments text)''')
    c.execute(
        '''CREATE TABLE event (chat_id text, action text, agent integer, time text, data text, start_time text)'''
    )
    c.execute(
        '''CREATE TABLE chat (chat_id text, scenario_id text, outcome text, agent_ids text, agent_types text,
        start_time text)'''
    )
    c.execute(
        '''CREATE TABLE scenario (scenario_id text, partner_type text, complete integer, active integer,
        PRIMARY KEY (scenario_id, partner_type))'''
    )

    conn.commit()
    conn.close()


def add_scenarios_to_db(db_file, scenario_db, systems):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for scenario in scenario_db.scenarios_list:
        sid = scenario.uuid
        for agent_type in systems.keys():
            c.execute('''INSERT INTO scenario VALUES (?,?, 0, 0)''', (sid, agent_type))

    conn.commit()
    conn.close()


def add_systems(config_dict, schema, lexicon, realizer):
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

    total_probs = 0.0
    systems = {HumanSystem.name(): HumanSystem()}
    pairing_probabilities = {}
    for (sys_name, info) in config_dict.iteritems():
        if "active" not in info.keys():
            warnings.warn("active status not specified for bot %s - assuming that bot is inactive." % sys_name)
        if info["active"]:
            type = info["type"]
            if type == SimpleSystem.name():
                model = SimpleSystem(lexicon, timed_session=True, realizer=realizer, consecutive_entity=False)
            elif type == NeuralSystem.name():
                path = info["path"]
                decoding = info["decoding"].split()
                model = NeuralSystem(schema, lexicon, path, decoding, timed_session=True, realizer=realizer, consecutive_entity=False)
            else:
                warnings.warn(
                    'Unrecognized model type in {} for configuration '
                    '{}. Ignoring configuration.'.format(info, sys_name))
                continue
            systems[sys_name] = model
            if 'prob' in info.keys():
                prob = float(info['prob'])
                pairing_probabilities[sys_name] = prob
                total_probs += prob

    if total_probs > 1.0:
        raise ValueError("Probabilities for active bots can't exceed 1.0.")
    if len(pairing_probabilities.keys()) != 0 and len(pairing_probabilities.keys()) != len(systems.keys()):
        remaining_prob = (1.0-total_probs)/(len(systems.keys()) - len(pairing_probabilities.keys()))
    else:
        remaining_prob = 1.0 / len(systems.keys())
    inactive_bots = set()
    for system_name in systems.keys():
        if system_name not in pairing_probabilities.keys():
            if remaining_prob == 0.0:
                inactive_bots.add(system_name)
            else:
                pairing_probabilities[system_name] = remaining_prob

    for sys_name in inactive_bots:
        systems.pop(sys_name, None)

    return systems, pairing_probabilities


def cleanup(flask_app):
    db_path = flask_app.config['user_params']['db']['location']
    transcript_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'transcripts.json')
    log_transcripts_to_json(flask_app.config['scenario_db'], db_path, transcript_path, None)


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
    add_lexicon_arguments(parser)
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

    app = create_app(debug=False, templates_dir=templates_dir)

    schema_path = args.schema_path

    if not os.path.exists(schema_path):
        raise ValueError("No schema file found at %s" % schema_path)

    schema = Schema(schema_path, domain=args.domain)
    # todo in the future would we want individual models to have different lexicons?
    lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words)
    if args.inverse_lexicon:
        realizer = InverseLexicon(schema, args.inverse_lexicon)
    else:
        realizer = None
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    app.config['scenario_db'] = scenario_db

    if 'models' not in params.keys():
        params['models'] = {}

    if 'quit_after' not in params.keys():
        params['quit_after'] = params['status_params']['chat']['num_seconds'] + 1

    if 'skip_chat_enabled' not in params.keys():
        params['skip_chat_enabled'] = False

    systems, pairing_probabilities = add_systems(params['models'], schema, lexicon, realizer)

    add_scenarios_to_db(db_file, scenario_db, systems)

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
    if 'icon' not in params.keys():
        app.config['task_icon'] = 'handshake.jpg'
    else:
        app.config['task_icon'] = params['icon']

    print "App setup complete"

    server = WSGIServer(('', args.port), app)
    atexit.register(cleanup, flask_app=app)
    server.serve_forever()
