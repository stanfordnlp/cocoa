# CoCoA (Collaborative Communicating Agents)

**CoCoA** is a dialogue research framework in Python, providing tools for
- **data collection** from a text-based chat interface by crowdsourcing (e.g. Amazon Mechanical Turk)
- **model development** with supports for rule-based bots, retrieval-based models, and neural models
- **human evaluation** by conversation partners and/or third-party evaluators
- **data analysis** including dialogue visualization, evaluation summary, and basic text and strategy analysis

The master branch currently supports the MutualFriends task from the paper [Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/pdf/1704.07130.pdf) (ACL 2017). More tasks are under development and will be added in future releases.

----------

## Installation
You must have Tensorflow 1.2 installed.
```
pip install -r requirements.txt
python setup.py develop
```

## Tasks
- **MutualFriends**: two agents, each with a private list of friends with multiple attributes (e.g. school, company), try to find their mutual friends through a conversation.
- **Craigslist Negotiation**: a buyer and a seller negotiate the price of an item for sale on [Craigslist](https://sfbay.craigslist.org/).
- **Deal or No Deal**: two agents negotiate to split a group of items with different points among them.  The items are books, hats and balls.
- **Open Movies**: Two agents have a conversation about movies, any discussion within the realm of movies is allowed.

## Main concepts/classes
### Schema and scenarios
A **dialogue** is always grounded in a **scenario** (structured context). A **schema** defines the structure of scenarios. You can think of the scenario as tables and the schema as the column definition. For example, a simple scenario that specifies the dialogue topic is

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
CoCoA is designed to be modular so that one can add their own task/modules easily, using `cocoa` as a package.
Below we use ```task``` as the custom task name (e.g. "negotiation").

### Data collection
#### Generate scenarios
We provide basic infrastructure to set up a website that pairs two users or a user and a bot to chat in a given scenario. The first step is to generate/write a ```.json``` schema file and then (randomly) generate a set of scenarios that the dialogue will be situated in. The scenario generation script is ```scripts/generate_scenarios.py```.

#### Setup the web server
The website pairs a user with another user or a bot (if available). A dialogue scenario is displayed and the two agents can chat with each other to complete the task until the time limit is reached. Users are then directed to a survey to rate their partners in terms of fluency, collaboration etc. All dialogue events are logged in a database.

Our server is built by [Flask](http://flask.pocoo.org/). The backend (```cocoa/web/main/backend.py```) maintains multiple systems (e.g. ```HumanSystem```, ```RulebasedSystem```, ```NeuralSystem```); when two agents are paired, they are put in two sessions and send/receive messages through the controller. See ```cocoa/web/views/``` for interacting with the front end. Task-specific templates are in ```task/web/templates```. The website config file specifies the time limit, systems/models etc.

To deploy the web server, run
```
cd <name-of-your-task>;
PYTHONPATH=. python web/chat_app.py --port <port> --config web/app_params.json --scenarios-path <path-to-scenarios> --output <output-dir>
```

To collect data from Amazon Mechanical Turk (AMT), workers should be directed to the link ```http://your-url:port/?mturk=1```. Workers will receive a Mturk code at the end of the survey to submit the HIT.

#### Dump the data
See ```scripts/web/dump_db.py```.

### Dialogue agents
To add an agent for a task, we need a corresponding system ```task/systems/agent_system.py``` and a session ```task/sessions/agent_session.py```.
Once an agent is implemented, we can let it self-play, i.e. chat with itself, using the script ```scripts/generate_dataset.py```.

### Model training and testing

### Human evaluation

Evaluations are done through AMT.

#### Evaluate context-response pairs

Two main classes are `EvalData` and `EvalTask`.
Take a look at `cocoa/turk/eval_data.py` and `cocoa/turk/task.py`.
They can be extended in `task/turk`.

We embed HTML in the AMT interface; see `cocoa/turk/templates`.

1. Generate system outputs and write to a JSON file.
```
outputs.json
|--[i]
|  |--"ex_id": unique id that identifies a context-reference pair
|  |--"prev_turns"
|  |--"reference"
|  |--"response"
```
For details of file formats, see `cocoa/turk/eval_data.py`.

2. Launch HITs! 

We use `boto`'s API. *NOTE*: always test on sandbox with `--debug` before launching.

```
cd negotiation;
PYTHONPATH=. python scripts/turk_eval.py --debug --aws-config <aws_credential> --question-template ../cocoa/turk/templates/question.html --overall-template ../cocoa/turk/templates/absolute_eval.html --instructions ../cocoa/turk/templates/instructions.html --num-eval 50 --num-assignments-per-hit 5 --system-outputs <system1_name> <system1_output> <system2_name> <system2_output>
```

3. Gather results.
```
cd negotiation;
PYTHONPATH=. python scripts/turk_eval.py --review --debug --aws-config <aws_credential> --question-template ../cocoa/turk/templates/question.html --overall-template ../cocoa/turk/templates/absolute_eval.html --instructions ../cocoa/turk/templates/instructions.html 
```
