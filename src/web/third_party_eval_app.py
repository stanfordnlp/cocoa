import argparse
import collections
import json
import os
import pprint
import random
import sqlite3
import uuid

from flask import Flask, render_template, request, jsonify, session, g
from gevent.wsgi import WSGIServer
from third_party_backend import BackendConnection

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=str, help="path to scenarios file")
parser.add_argument("--examples", type=str, help="path to examples file")
parser.add_argument("--port", type=int, default=5000, help="port to launch app on")
parser.add_argument("--num-evals-per-worker", type=int, help="number of evaluations per worker before redeeming code")
parser.add_argument("--num-evals-per-dialogue", type=int, help="number of evaluations per dialogue ")
args = parser.parse_args()



def init_database(db_path):
    """
    Initalize database
    :param db_path:  Path to db
    :return:
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE Responses (dialogue_id text, scenario_id text, agent_mapping text, user_id text, agent_id integer, humanlike text, correct text,
              cooperative text, fluent text, humanlike_text text, correct_text text, cooperative_text text, fluent_text text)""")

    c.execute("""CREATE TABLE ActiveDialogues (dialogue_id text unique, scenario_id text, events text, column_names text, agent_mapping text, agent0_kb text, agent1_kb text, num_agent0_evals integer, num_agent1_evals integer)""")

    c.execute("""CREATE TABLE CompletedDialogues(dialogue_id text, scenario_id text, agent_mapping text, num_agent0_evals integer, num_agent1_evals integer, timestamp text)""")

    c.execute("""CREATE TABLE ActiveUsers (user_id text unique, agent0_dialogues_evaluated text, agent1_dialogues_evaluated text, num_evals_completed integer, timestamp text)""")

    c.execute("""CREATE TABLE CompletedUsers (user_id text, mturk_code text, timestep text, num_evals_completed integer)""")
    conn.commit()
    conn.close()


def init_dialogues(db_path):
    """
    Initialize all dialogues if not previously initialized
    :param db_path: Path to db
    :return:
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Load examples containing UUIDs for scenarios
    examples_file = args.examples
    with open(examples_file, "r") as f:
        examples = json.load(f)

    # Load scenarios containing KB info
    scenarios_json = args.scenarios
    with open(scenarios_json, "r") as f:
        scenarios_info = json.load(f)

    # Form mapping from uuid -> scenario
    uuid_to_scenario = collections.defaultdict(dict)
    for scenario in scenarios_info:
        scenario_id = scenario["uuid"]
        uuid_to_scenario[scenario_id] = scenario


    for ex in examples:
        scenario_id = ex["scenario_uuid"]
        scenario = uuid_to_scenario[scenario_id]
        kbs = scenario["kbs"]
        agent0_kb = kbs[0]
        agent1_kb = kbs[1]
        column_names = agent0_kb[0].keys()

        # Get events for example
        events = ex["events"]

        # Filter to only include message data
        msg_events = []
        # Keep track of those agents that are actually participating
        agents_present = set()
        for event in events:
            agents_present.add(event["agent"])
            if event["action"] == "message":
                msg_events.append(event)
            elif event["action"] == "select":
                # Modify format of selection data
                new_data = "SELECT("
                for attr in event["data"].iteritems():
                    new_data += attr[1] + ", "
                new_data = new_data.strip()
                new_data = new_data.strip(",")
                new_data += ")"
                event["data"] = new_data
                msg_events.append(event)


        try:
            c.execute("SELECT dialogue_id FROM CompletedDialogues")
            d_ids = set([d_id[0] for d_id in c.fetchall()])
        except:
            print "No Completed dialogues table"

        dialogue_id = ex["uuid"]
        agents_mapping = ex["agents"]
        if len(agents_present) == 2:
            if dialogue_id not in d_ids:
                c.execute("""INSERT OR IGNORE INTO ActiveDialogues VALUES (?,?,?,?,?,?,?,?,?) """,
                    (dialogue_id, scenario_id, json.dumps(msg_events), json.dumps(column_names), json.dumps(agents_mapping), json.dumps(agent0_kb),
                    json.dumps(agent1_kb), 0, 0))
            else:
                print "Skipping dialogue id: ", dialogue_id
        else:
            print "Skipped: ", dialogue_id

    conn.commit()
    conn.close()


def get_backend():
    backend = getattr(g, '_backend', None)
    if backend is None:
        print "Creating backend..."
        backend = g._backend = BackendConnection(app.config["db_path"])
    return backend


def set_or_get_userid():
    if "sid" in session and session["sid"]:
        return userid()
    if "sid" not in session or not session["sid"]:
        print "New user created!"
        session["sid"] = str(uuid.uuid4())[:12]
    print "Session ID: ", session["sid"]
    return session["sid"]


def userid():
    return session["sid"]


@app.route("/submit", methods=["GET", "POST"])
def handle_submit():
    results = request.get_json()

    backend = get_backend()
    print "USER ID: ", userid()
    print "Scenario ID: ", results["scenario_id"]
    print "Dialogue ID: ", results["dialogue_id"]

    backend.submit_task(userid(), results["scenario_id"], results, app)


    return jsonify(result={"status": 200})


@app.route('/')
@app.route('/index')
def index():
    _ = set_or_get_userid()
    backend = get_backend()
    backend.create_user_if_necessary(userid())
    num_evals_completed = backend.get_num_evals_completed(userid())
    if num_evals_completed < app.config["num_evals_per_worker"]:
        dialogue = backend.get_dialogue(userid(), app)
        # If no dialogue found
        if dialogue is None:
            return render_template("third_party_eval_finished.html",
                               mturk_code=str(random.randint(0, 10000)),
                               finished_message="You have exceeded the allowed number of dialogues to be evaluated and hence can no longer do these HITs! If you believe this"
                                                " message was given in error, please contact us.")
        else:
            return render_template("third_party_eval.html",
                                   dialogue=json.loads(dialogue["events"]),
                                   agent_id=dialogue["agent_id"],
                                   dialogue_id=dialogue["dialogue_id"],
                                   scenario_id=dialogue["scenario_id"],
                                   column_names=json.loads(dialogue["column_names"]),
                                   kb=json.loads(dialogue["kb"]),
                                   )
    else:
        mturk_code = backend.get_finished_info(userid())
        return render_template("third_party_eval_finished.html",
                               mturk_code=mturk_code,
                               finished_message="YOU HAVE FINISHED THE HIT!")


db_path = "third_party_eval.db"
# Init DB
if os.path.exists(db_path):
    print "DB already exists"
else:
    print "Creating new DB"
    init_database(db_path)

init_dialogues(db_path)


# Launch server
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
app.config["num_evals_per_worker"] = args.num_evals_per_worker
app.config["num_evals_per_dialogue"] = args.num_evals_per_dialogue
app.config["db_path"] = db_path

server = WSGIServer(('', args.port), app)
server.serve_forever()