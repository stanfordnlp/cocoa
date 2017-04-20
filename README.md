# CoCoA (Collaborative Communicating Agents)                                                    
We build Collaborative Communicating Agents (CoCoA) that collaborate with humans through natural language communication. We focus on the symmetric collaborative dialogue setting, where two agents, each with private knowledge, must communicate to achieve a common goal.

This branch contains code for the paper "Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings".

## Dependencies
The following Python packages are required.
- General: Python 2.7, NumPy 1.11, Tensorflow r0.12
- Lexicon: fuzzywuzzy, editdistance
- Web server: flask, gevent, sqlite3

You can also use the docker image `hhexiy/cocoa:0.2`.

## Data collection
### Scenario generation
Generate the schema:
```
mkdir data/cache
python src/scripts/generate_schema.py --schema-path data/schema.json --cache-path data/cache
```
Generate scenarios from the schema:
```
python src/scripts/generate_scenarios.py --schema-path data/schema.json --scenarios-path data/scenarios.json --num-scenarios 500 --random-attributes --random-items --alphas 0.3 1 3 
```
Some of the parameters used;
- `--random-attributes`: For each scenario, select a random number and subset of attributes from the provided schema (if not provided, the script always includes all the attributes in the schema in each scenario)
- `--min-attributes`: If `--random-attributes` is provided, specifies the minimum number of attributes to be chosen per scenario
- `--max-attributes`: If `--random-attributes` is provided, specifies the maximum number of attributes to be chosen per scenario
- `--random-items`: For each scenario, select a random number of items. If this parameter is not provided, the `--num-items` argument can be used to set the number of items per scenario.
- `--min-items`: If `--random-items` is provided, specifies the minimum number of items to be chosen per scenario
- `--max-items`: If `--random-items` is provided, specifies the maximum number of items to be chosen per scenario
- `--alphas`: A list of alpha values to select from for each attribute in each scenario (the alpha value specifies the uniformness of the distribution that the attribute values are sampled from; higher alpha values imply that attribute values will be more unique.)
      This parameter is only used if `--random-attributes` is provided. 

### Set up the web server
The website provides a basic infrastructure to allow to users to chat with each other, or allow a user to randomly chat with a bot.
 When users first enter the website, they wait until they are paired up with another user (or a bot), and are then allowed to chat for a specified time, to complete the MutualFriends task, before the task times out. 
 Users can then be directed to either a survey or a 'finished' page with a Mechanical Turk code.
 To set up the website, follow these instructions:
 
 1. Modify the website parameters in `data/web/app_params.json`:
     1. `status_params`: Specifies how long the user is allowed to stay in different "states" before the task times out. These parameters don't need to be modified for the MutualFriends task.
     2. `templates_dir` and `instructions`: Modify these paths to point to `src/web/templates` and `data/web/mutual-friends-instructions.html`
     3. `models`: You can specify any number of bots that users can chat with here. Currently, only the simple rule-based bot and the neural bot (backed by DynoNet) are supported. `models` must be a dictionary of unique bot names mapping to a bot definition. The bot definition must be a dictionary with the following fields:
         1. `active`: true or false, depending on whether the bot should be active or not when the website is started. If false, the bot will not be loaded and users will not be paired with it.
         2. `type`: "simple" or "neural", depending on the bot type
         3. `path`: (for neural bots only) the location of the model checkpoint
         4. `decoding`: (for neural bots only) The type of decoding to use for the DynoNet (see its documentation)
         5. `prob`: (optional) The probability that a human is paired with the bot. By default, this probability is 1/(N+1), where N is the number of active bots, since humans can be paired with other humans on the website by default. To completely disable human-human pairings, you can manually set the probabilities of all active bots such that they sum to 1.0. In general, if the probability of any one bot is specified, the probability of all 
       other bots with unspecified probabilities (and of human pairings) is computed by dividing the remaining probability equally among all partner types. For example, if you specify two active bots A and B, with prob=0.4 for A,
       and unspecified for B, then the remaining probability of 0.6 is divided equally among B and human, such that p(A) = 0.4, P(B) = P(human) = 0.3.
 2. Run the website: ```python src/web/start_app.py --port 5000 --schema-path data/friends-schema.json --scenarios-path data/scenarios.json --config data/web/app_params.json```
 3. By default, the website code ensures that at least one complete chat is collected per scenario, per partner type (assuming that there is enough traffic to the website). So, for example, if you provide 200 scenarios
  and 3 bot types (for a total of 4 partner types, including humans), the website will attempt to collect at least 200*4=800 completed chats.

## Model
Main code for the Dynamic Knowledge Graph Network (DynoNet) is in `src/model`. Specifically,
- `graph.py`: Data structure for the knowledge base.
- `graph_embedder.py`: Compute the graph embedding given the graph structure (and utterance embeddings).
- `encdec.py`: Encoder-decoder with attention and copy mechanism over the graph embeddings.
Hyperparameters can be found in `add_<module>_arguments` in different module files.

### Training
The script `main.py` does data preprocessing, entity linking, and training (testing is performed at the end of each epoch on the dev set).
The output models are saved in `checkpoint` (vocabulary, config file and the most recent 5 models) and `checkpoint-best` (the best model evaluated on the dev set). 
The following command trains a DynoNet, for other models see the configuration on [Codalab](https://worksheets.codalab.org/worksheets/0xc757f29f5c794e5eb7bfa8ca9c945573/).
```
PYTHONPATH=. python src/main.py --schema-path data/schema.json --scenarios-path data/scenarios.json
--train-examples-paths data/train.json --test-examples-paths data/dev.json --stop-words data/common_words.txt
--min-epochs 10 --checkpoint checkpoint --rnn-type lstm --learning-rate 0.5 --optimizer adagrad
--print-every 50 --model attn-copy-encdec --gpu 1 --rnn-size 100 --grad-clip 0 --num-items 12
--batch-size 32 --stats-file stats.json --entity-encoding-form type --entity-decoding-form type
--node-embed-in-rnn-inputs --msg-aggregation max --word-embed-size 100 --node-embed-size 50
--entity-hist-len -1 --learned-utterance-decay
```

## Evaluation
### Test set loss and response generation
Compute the cross entropy and responses generated given ground truth dialogue history:
```
PYTHONPATH=. python src/main.py --schema-path data/schema.json --scenarios-path data/scenarios.json
--stop-words data/common_words.txt --init-from checkpoint --test-examples-paths data/test.json
--test --batch-size 32 --best --stats-file stats.json --decoding sample 0 --verbose
```

### Bot-bot chat
Dialogue modesl are launched by a system in one ore more sessions (see `src/basic/systems` and `src/basic/sessions`).
Generate bot-bot chat:
```
PYTHONPATH=. python src/scripts/generate_dataset.py --max-turns 46 --schema-path data/schema.json
--scenarios-path data/scenarios.json --stop-words data/common_words.txt --test-examples-paths chat.json
--train-max-examples 0 --agents neural neural --model-path model/checkpoint --decoding sample 0.5 select
```
To use the rule-based bot, set `--agents simple simple`.

### Analysis
To compute various statistics given a chat transcript, run
```
PYTHONPATH=. python src/scripts/get_data_statistics.py --transcripts chat.json --schema-path data/schema.json
--scenarios-path data/scenarios.json --stop-words data/common_words.txt --stats-output dialogue_stats.json
```
The results are saved in `dialogue_stats.json`.

### Visualization
There are two ways to visualize a chat transcript (in json).
To output a plain HTML file that contains all chats, run
```
PYTHONPATH=. python src/scripts/visualize_data.py --transcripts chat.json --schema-path data/schema.json
--scenarios-path data/scenarios.json --html-output chat.html
```
To display metadata of each chat (e.g., outcome, number of attributes) and perform filter/search, run
```
PYTHONPATH=. python src/scripts/visualize_data.py --transcripts chat.json --schema-path data/schema.json
--scenarios-path data/scenarios.json --html-output chat.html --html-output . --viewer-mode
```
Each chat is then saved as a separate HTML files in `chat_htmls`. To view the chats, open `chat_viewer/chat.html` in a browser.










