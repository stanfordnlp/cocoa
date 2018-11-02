# CoCoA (Collaborative Communicating Agents)

**CoCoA** is a dialogue framework written in Python, providing tools for
data collection through a text-based chat interface and
model development in PyTorch (largely based on OpenNMT).

This repo contains code for the following tasks:
- **MutualFriends**: two agents, each with a private list of friends with multiple attributes (e.g. school, company), try to find their mutual friends through a conversation.
- **CraigslistBargain**: a buyer and a seller negotiate the price of an item for sale on [Craigslist](https://sfbay.craigslist.org/).
- **DealOrNoDeal**: two agents negotiate to split a group of items with different points among them.  The items are books, hats and balls.

**Papers**:
- [Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/pdf/1704.07130.pdf).
He He, Anusha Balakrishnan, Mihail Eric and Percy Liang.
Association for Computational Linguistics (ACL), 2017.
- [Decoupling Strategy and Generation in Negotiation Dialogues](https://arxiv.org/abs/1808.09637).
He He, Derek Chen, Anusha Balakrishnan and Percy Liang.
Empirical Methods in Natural Language Processing (EMNLP), 2018.

**Note**:
We have not fully integrated the MutualFriends task with the `cocoa` package.
For now please refer to the `mutualfriends` branch for the ACL 2017 paper.

----------
## Installation
**Dependencies**: Python 2.7, PyTorch 0.4.

**NOTE**: MutualFriends still depends on Tensorflow 1.2 and uses different leanring modules. See details on the `mutualfriends` branch.

```
pip install -r requirements.txt
python setup.py develop
```

## Main concepts/classes
### Schema and scenarios
A **dialogue** is grounded in a **scenario**.
A **schema** defines the structure of scenarios. For example, a simple scenario that specifies the dialogue topic is

| Topic      | 
| -------- | 
| Artificial Intelligence  | 

and its schema (in JSON) is
```
{
    "attributes": [
        "value_type": "topic",
        "name": "Topic"
    ]
}
```
### Systems and sessions
A dialogue **agent** is instantiated in a **session** which receives and sends messages. A **system** is used to create multiple sessions (that may run in parallel) of a specific agent type. For example, ```system = NeuralSystem(model)``` loads a trained model and ```system.new_session()``` is called to create a new session whenever a human user is available to chat.

### Events and controllers
A dialogue **controller** takes two sessions and have them send/receive **events** until the task is finished or terminated. The most common event is ```message```, which sends some text data. There are also task-related events, such as ```select``` in MutualFriends, which sends the selected item. 

### Examples and datasets
A dialogue is represented as an **example** which has a scenario, a series of events, and some metadata (e.g. example id). Examples can be read from / write to a JSON file in the following structure:
```
examples.json
|--[i]
|  |--"uuid": "<uuid>"
|  |--"scenario_uuid": "<uuid>"
|  |--"scenario": "{scenario dict}"
|  |--"agents": {0: "agent type", 1: "agent type"}
|  |--"outcome": {"reward": R}
|  |--"events"
|     |--[j]
|        |--"action": "action"
|        |--"data": "event data"
|        |--"agent": agent_id
|        |--"time": "event sent time"
```
A **dataset** reads in training and testing examples from JSON files.

## Code organization
CoCoA is designed to be modular so that one can add their own task/modules easily.
All tasks depend on the `cocoa` pacakge.
See documentation in the task folder for task-specific details. 

### Data collection
We provide basic infrastructure (see `cocoa.web`) to set up a website that pairs two users or a user and a bot to chat in a given scenario.

#### Generate scenarios
The first step is to create a ```.json``` schema file and then (randomly) generate a set of scenarios that the dialogue will be situated in.

#### <a name=web>Setup the web server</a>
The website pairs a user with another user or a bot (if available). A dialogue scenario is displayed and the two agents can chat with each other.
Users are then directed to a survey to rate their partners (optional).
All dialogue events are logged in a SQL database.

Our server is built by [Flask](http://flask.pocoo.org/).
The backend (```cocoa/web/main/backend.py```) contains code for pairing, logging, dialogue quality check.
The frontend code is in ```task/web/templates```.

To deploy the web server, run
```
cd <name-of-your-task>;
PYTHONPATH=. python web/chat_app.py --port <port> --config web/app_params.json --schema-path <path-to-schema> --scenarios-path <path-to-scenarios> --output <output-dir>
```
- Data and log will be saved in `<output-dir>`. **Important**: note that this will delete everything in `<output-dir>` if it's not empty.
- `--num-scenarios`: total number of scenarios to sample from. Each scenario will have `num_HITs / num_scenarios` chats.
You can also specify ratios of number of chats for each system in the config file.
Note that the final result will be an approximation of these numbers due to concurrent database calls.

To collect data from Amazon Mechanical Turk (AMT), workers should be directed to the link ```http://your-url:<port>/?mturk=1```.
`?mturk=1` makes sure that workers will receive a Mturk code at the end of the task to submit the HIT.

#### <a name=visualize>Dump the database</a>
Dump data from the SQL database to a JSON file (see [Examples and datasets](#examples-and-datasets) for the JSON structure).
```
cd <name-of-your-task>;
PYTHONPATH=. python ../scripts/web/dump_db.py --db <output-dir>/chat_state.db --output <output-dir>/transcripts/transcripts.json --surveys <output-dir>/transcripts/surveys.json --schema <path-to-schema> --scenarios-path <path-to-scenarios> 
```
Render JSON transcript to HTML:
```
PYTHONPATH=. python ../scripts/visualize_transcripts.py --dialogue-transcripts <path-to-json-transcript> --html-output <path-to-output-html-file> --css-file ../chat_viewer/css/my.css
```
Other options for HTML visualization:
- `--survey-transcripts`: path to `survey.json` if survey is enabled during data collection.
- `--survey-only`: only visualize dialgoues with submitted surveys.
- `--summary`: statistics of the dialogues.

### Dialogue agents
To add an agent for a task, you need to implement a system ```<name-of-your-task>/systems/<agent-name>_system.py```
and a session ```<name-of-your-task>/sessions/<agent-name>_session.py```.

### Model training and testing
See documentation in the under each task (e.g., `./craigslistbargain`).

### Evaluation
To deploy bots to the web interface, add the `"models"` field in the website config file,
e.g.
```
"models": {
    "rulebased": {
        "active": true,
        "type": "rulebased",
    }
}
```
See also [set up the web server](#web).
