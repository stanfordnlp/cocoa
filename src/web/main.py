import collections
import json
import random

from flask import Flask, render_template


app = Flask(__name__)


# Load scenarios containing KB info
scenarios_json = "/Users/mihaileric/Documents/Research/game-dialogue/src/web/static/friends-scenarios-large.json"
with open(scenarios_json, "r") as f:
    scenarios_info = json.load(f)

uuid_to_scenario = collections.defaultdict(dict)
for scenario in scenarios_info:
    uuid = scenario["uuid"]
    uuid_to_scenario[uuid] = scenario

# Load examples containing UUIDs for scenarios
examples_file = "/Users/mihaileric/Documents/Research/game-dialogue/src/web/static/transcripts.json"
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


@app.route('/')
@app.route('/index')
def index():
    # Sample random example
    ex_idx = random.sample(range(num_examples), 1)
    ex = idx_to_examples[ex_idx[0]]
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
    msg_events = {idx: event for idx, event in enumerate(events)
                  if event['action'] == "message"}

    # Will want to annotate by span of tokens
    return render_template("single_task_lexicon.html",
                           column_names=column_names,
                           kb1=agent1_kb,
                           kb2=agent2_kb,
                           dialogue=msg_events,
                           entities=entities)


# Launch server
app.run()