import argparse
import collections
import json
import pprint
import random

from flask import Flask, render_template, request, jsonify, session


app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=str, help="path to scenarios file")
parser.add_argument("--examples", type=str, help="path to examples file")
args = parser.parse_args()


# Load scenarios containing KB info
scenarios_json = args.scenarios
with open(scenarios_json, "r") as f:
    scenarios_info = json.load(f)

uuid_to_scenario = collections.defaultdict(dict)
for scenario in scenarios_info:
    uuid = scenario["uuid"]
    uuid_to_scenario[uuid] = scenario

# Load examples containing UUIDs for scenarios
examples_file = args.examples
with open(examples_file, "r") as f:
    examples = json.load(f)

idx_to_examples = {idx: ex for idx, ex in enumerate(examples)}
num_examples = len(examples)
seen_examples = set()


def get_all_entities(kbs):
    entities = set()
    for kb in kbs:
        for row in kb:
            for _, value in row.iteritems():
                entities.add(value)

    entities = list(entities)
    return sorted(entities)


annotated_examples = []

@app.route("/submit", methods=["GET", "POST"])
def handle_submit():
    global annotated_examples
    global count
    count += 1
    current_example = session["current_example"]

    results = request.get_json()
    for idx, annotations in results.iteritems():
        idx = int(idx)

        event_annotations = []
        for a in annotations:
            # (span start, span end, text, entity)
            span_text_entity = tuple(a[1:])
            event_annotations.append(span_text_entity)

        current_example["events"][idx]["entityAnnotation"] = event_annotations

    pprint.pprint(current_example["events"])
    annotated_examples.append(current_example)

    # Dump if annotated
    if count % 10 == 0:
        with open("annotated_examples.json", "w") as f:
            print "WRITING ANNOTATIONS TO DISK"
            json.dump(annotated_examples, f)

    return jsonify(result={"status": 200})


count = 0
@app.route('/')
@app.route('/index')
def index():
    print "GLOBAL Count: ", count
    # Sample random example
    while True:
        ex_idx = random.sample(range(num_examples), 1)
        if ex_idx[0] not in seen_examples:
            break
    ex = idx_to_examples[ex_idx[0]]
    seen_examples.add(ex_idx[0])

    uuid = ex["scenario_uuid"]
    scenario = uuid_to_scenario[uuid]
    kbs = scenario["kbs"]
    agent1_kb = kbs[0]
    agent2_kb = kbs[1]
    entities = get_all_entities(kbs)
    column_names = agent1_kb[0].keys()

    # Get events for example
    events = ex["events"]
    idx_to_event = {idx: event for idx, event in enumerate(events)}
    # Filter to only include message data
    msg_events = {}
    for idx, event in enumerate(events):
        if event["action"] == "message":
            msg_events[idx] = event

    if not "count" in session:
        session["count"] = count
    else:
        session["count"] += 1
        print "Session count in else: ", session["count"]

    # Keep track of current examples being annotated
    session["current_example"] = ex

    # Will want to annotate by span of tokens
    return render_template("single_task_lexicon.html",
                           column_names=column_names,
                           kb1=agent1_kb,
                           kb2=agent2_kb,
                           dialogue=msg_events,
                           entities=entities)


# Launch server
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
app.run(debug=True)