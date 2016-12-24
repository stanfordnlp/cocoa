import argparse
import collections
import json
import os
import pprint
import random
import sqlite3

from flask import Flask, render_template, request, jsonify, session
from gevent.wsgi import WSGIServer

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=str, help="path to scenarios file")
parser.add_argument("--examples", type=str, help="path to examples file")
args = parser.parse_args()

def init_database(db_file):
    """
    Initalize database
    :param db_file:  Path to db
    :return:
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""CREATE TABLE Responses (dialogue_num integer, humanlike_0 text, effective_0 text, strategic_0 text)""")
    conn.commit()
    conn.close()


# Load examples containing UUIDs for scenarios
examples_file = args.examples
with open(examples_file, "r") as f:
    examples = json.load(f)

idx_to_examples = {idx: ex for idx, ex in enumerate(examples)}
num_examples = len(examples)
seen_examples = set()

db_path = "third_party_eval.db"

# Init DB
if os.path.exists(db_path):
    print "DB already exists"
else:
    print "Creating new DB"
    init_database(db_path)


@app.route("/submit", methods=["GET", "POST"])
def handle_submit():
    backend = sqlite3.connect(db_path)
    results = request.get_json()
    
    pprint.pprint(results)

    # Insert into dialogue
    with backend:
        c = backend.cursor()
        c.execute("INSERT INTO Responses VALUES (?, ?, ?, ?)", (results["name"], results["humanlike_0"],
                                                                results["effective_0"], results["strategic_0"]))
        backend.commit()

    return jsonify(result={"status": 200})


@app.route('/')
@app.route('/index')
def index():
    # Sample random example
    while True:
        ex_idx = random.sample(range(num_examples), 1)
        if ex_idx[0] not in seen_examples:
            break
    ex = idx_to_examples[ex_idx[0]]
    seen_examples.add(ex_idx[0])

    # Get events for example
    events = ex["events"]

    # Filter to only include message data
    msg_events = {}
    for idx, event in enumerate(events):
        if event["action"] == "message":
            msg_events[idx] = event


    # Keep track of current examples being annotated
    session["current_example"] = ex

    # Will want to annotate by span of tokens
    return render_template("third_party_eval.html",
                           dialogue=msg_events,
                           name=ex["scenario_uuid"]
                           )


# Launch server
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

server = WSGIServer(('', 5000), app)
server.serve_forever()