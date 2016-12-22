import argparse
import collections
import json
import pprint
import random

from flask import Flask, render_template, request, jsonify, session
from gevent.wsgi import WSGIServer

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=str, help="path to scenarios file")
parser.add_argument("--examples", type=str, help="path to examples file")
args = parser.parse_args()


# Load examples containing UUIDs for scenarios
examples_file = args.examples
with open(examples_file, "r") as f:
    examples = json.load(f)

idx_to_examples = {idx: ex for idx, ex in enumerate(examples)}
num_examples = len(examples)
seen_examples = set()



@app.route("/submit", methods=["GET", "POST"])
def handle_submit():
    global count
    count += 1
    current_example = session["current_example"]

    results = request.get_json()

    pprint.pprint(current_example["events"])


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
                           )


# Launch server
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

server = WSGIServer(('', 5000), app)
server.serve_forever()