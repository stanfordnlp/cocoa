

**CoCoA** is a dialogue research framework in Python, providing tools for
- **data collection** from a text-based chat interface by crowdsourcing (e.g. Amazon Mechanical Turk)
- **model development** with supports for rule-based bots, retrieval-based models, and neural models
- **human evaluation** by conversation partners and/or third-party evaluators
- **data analysis** including dialogue visualization, evaluation summary, and basic text and strategy analysis

The master branch currently supports the MutualFriends task from the paper [Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/pdf/1704.07130.pdf) (ACL 2017). More tasks are under development and will be added in future releases.

----------

[TOC]

## Tasks
- **MutualFriends**: two agents, each with a private list of friends with multiple attributes (e.g. school, company), try to find their mutual friends through a conversation.
- **Craigslist Negotiation**: a buyer and a seller negotiate the price of an item for sale on [Craigslist](https://sfbay.craigslist.org/).

## Main concepts/classes
### Schema and scenarios
A **dialogue** is always grounded in a **scenario** (structured context). A **schema** defines the structure of scenarios. You can think of the scenario as tables and the schema as the column definition. For example, a simple scenario that specifies the dialogue topic is
| Topic      | 
| :-------- | 
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
CoCoA is designed to be modular so that one can plug in their own task/modules easily. Below we use ```task``` as the custom task name (e.g. "mutualfriends") in modules or paths.

### Data collection
#### Generate scenarios
We provide basic infrastructure to set up a website that pairs two users or a user and a bot to chat in a given scenario. The first step is to generate/write a ```.json``` schema file and then (randomly) generate a set of scenarios that the dialogue will be situated in. The scenario generation script should be ```src/scripts/task/generate_scenarios.py```.

#### Setup the web server
The website pairs a user with another user or a bot (if available). A dialogue scenario is displayed and the two agents can chat with each other to complete the task until the time limit is reached. Users are then directed to a survey to rate their partners in terms of fluency, collaboration etc. All dialogue events are logged in a database.

Our server is built by [Flask](http://flask.pocoo.org/). The backend (```src/web/main/backend.py```) maintains multiple systems (e.g. ```HumanSystem```, ```RulebasedSystem```, ```NeuralSystem```); when two agents are paired, they are put in two sessions and send/receive messages through the controller. See ```src/web/main/routes.py``` for interacting with the front end. Task-specific templates are in ```src/web/templates/task```. The website config file is ```data/web/task/app_params.json```, specifying the time limit, systems/models etc.

To deploy the web server, run
```
python src/web/main/start_app.py --port <port> --config <path-to-config> --scenario-path <path-to-scenarios>
```

To collect data from Amazon Mechanical Turk, simply provide the address ```http://your-url:port/?mturk=1``` in your HIT. Workers will receive a Mturk code at the end of the survey to submit the HIT.

#### Dump the data
See ```src/web/dump_events_to_json.py```.
### Dialogue agents
To add an agent for a task, we need a corresponding system ```src/systems/task/agent_system.py``` and a session ```src/sessions/task/agent_session.py```.
Once an agent is implemented, we can let it self-play, i.e. chat with itself, using the script ```src/scripts/generate_dataset.py```.
### Model training and testing
### Human evaluation


