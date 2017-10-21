import argparse
from collections import defaultdict
import json
import sqlite3
from datetime import datetime
import os
import shutil
import warnings
import atexit
import sys
from gevent.pywsgi import WSGIServer

from cocoa.core.scenario_db import add_scenario_arguments, ScenarioDB
from cocoa.core.schema import Schema
from cocoa.core.util import read_json
from cocoa.systems.human_system import HumanSystem
from cocoa.web.main.logger import WebLogger
#from cocoa.web.dump_events_to_json import log_transcripts_to_json, log_surveys_to_json

from core.scenario import Scenario
from systems import get_system, add_system_arguments
from main.db_reader import DatabaseReader
from main.backend import DatabaseManager

from web.main.backend import Backend
from flask import g
from flask import Flask, current_app
from flask_socketio import SocketIO
socketio = SocketIO()


__author__ = 'derekchen'

DB_FILE_NAME = 'chat_state.db'
LOG_FILE_NAME = 'log.out'
ERROR_LOG_FILE_NAME = 'error_log.out'
TRANSCRIPTS_DIR = 'transcripts'

def close_connection(exception):
    backend = getattr(g, '_backend', None)
    if backend is not None:
        backend.close()


def create_app(debug=False, templates_dir='templates'):
    """Create an application."""

    app = Flask(__name__, template_folder=os.path.abspath(templates_dir))
    app.debug = debug
    app.config['SECRET_KEY'] = 'gjr39dkjn344_!67#'
    app.config['PROPAGATE_EXCEPTIONS'] = True

    from web.views.action import action
    from cocoa.web.views.chat import chat
    app.register_blueprint(chat)
    app.register_blueprint(action)

    app.teardown_appcontext_funcs = [close_connection]

    socketio.init_app(app)
    return app
######################

def add_website_arguments(parser):
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host IP address to run app on. Defaults to localhost.')
    parser.add_argument('--config', type=str, default='app_params.json',
                        help='Path to JSON file containing configurations for website')
    parser.add_argument('--output', type=str,
                        default="web/output/{}".format(datetime.now().strftime("%Y-%m-%d")),
                        help='Name of directory for storing website output (debug and error logs, chats, '
                             'and database). Defaults to a web_output/current_date, with the current date formatted as '
                             '%%Y-%%m-%%d. '
                             'If the provided directory exists, all data in it is overwritten unless the '
                             '--reuse parameter is provided.')
    parser.add_argument('--reuse', action='store_true', help='If provided, reuses the existing database file in the '
                                                             'output directory.')


def add_systems(args, config_dict, schema, debug=False):
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
    timed = False if debug else True
    for (sys_name, info) in config_dict.iteritems():
        if "active" not in info.keys():
            warnings.warn("active status not specified for bot %s - assuming that bot is inactive." % sys_name)
        if info["active"]:
            name = info["type"]
            try:
                model = get_system(name, args, schema=schema, timed=timed)
            except ValueError:
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
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    DatabaseReader.dump_chats(cursor, flask_app.config['scenario_db'], transcript_path)
    if flask_app.config['user_params']['end_survey'] == 1:
        surveys_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'surveys.json')
        DatabaseReader.dump_surveys(cursor, surveys_path)
    conn.close()

def init(output_dir, reuse=False):
    db_file = os.path.join(output_dir, DB_FILE_NAME)
    log_file = os.path.join(output_dir, LOG_FILE_NAME)
    error_log_file = os.path.join(output_dir, ERROR_LOG_FILE_NAME)
    transcripts_dir = os.path.join(output_dir, TRANSCRIPTS_DIR)
    # TODO: don't remove everything
    if not reuse:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        db = DatabaseManager.init_database(db_file)

        if os.path.exists(transcripts_dir):
            shutil.rmtree(transcripts_dir)
        os.makedirs(transcripts_dir)
    else:
        db = DatabaseManager(db_file)

    return db, log_file, error_log_file, transcripts_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scenarios', type=int)
    add_website_arguments(parser)
    add_scenario_arguments(parser)
    add_system_arguments(parser)
    args = parser.parse_args()

    params_file = args.config
    with open(params_file) as fin:
        params = json.load(fin)

    db, log_file, error_log_file, transcripts_dir = init(args.output, args.reuse)
    error_log_file = open(error_log_file, 'w')

    WebLogger.initialize(log_file)
    params['db'] = {}
    params['db']['location'] = db.db_file
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

    app = create_app(debug=params['debug'], templates_dir=templates_dir)

    schema_path = args.schema_path

    if not os.path.exists(schema_path):
        raise ValueError("No schema file found at %s" % schema_path)

    schema = Schema(schema_path)
    scenarios = read_json(args.scenarios_path)
    if args.num_scenarios is not None:
        scenarios = scenarios[:args.num_scenarios]
    scenario_db = ScenarioDB.from_dict(schema, scenarios, Scenario)
    app.config['scenario_db'] = scenario_db

    if 'models' not in params.keys():
        params['models'] = {}

    if 'quit_after' not in params.keys():
        params['quit_after'] = params['status_params']['chat']['num_seconds'] + 1

    if 'skip_chat_enabled' not in params.keys():
        params['skip_chat_enabled'] = False

    if 'end_survey' not in params.keys() :
        params['end_survey'] = 0

    systems, pairing_probabilities = add_systems(args, params['models'], schema, params['debug'])
    db.add_scenarios(scenario_db, systems, update=args.reuse)

    app.config['systems'] = systems
    app.config['sessions'] = defaultdict(None)
    app.config['pairing_probabilities'] = pairing_probabilities
    app.config['num_chats_per_scenario'] = params.get('num_chats_per_scenario', {k: 1 for k in systems})
    for k in systems:
        assert k in app.config['num_chats_per_scenario']
    app.config['schema'] = schema
    app.config['user_params'] = params
    app.config['controller_map'] = defaultdict(None)
    app.config['instructions'] = instructions
    app.config['task_title'] = params['task_title']


    if 'icon' not in params.keys():
        app.config['task_icon'] = 'handshake.jpg'
    else:
        app.config['task_icon'] = params['icon']

    print "App setup complete"

    server = WSGIServer(('', args.port), app, log=WebLogger.get_logger(), error_log=error_log_file)
    atexit.register(cleanup, flask_app=app)
    server.serve_forever()
