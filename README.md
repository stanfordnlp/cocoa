# CoCoA (Collaborative Communicating Agents)                                                    
We build Collaborative Communicating Agents (CoCoA) that collaborate with humans through natural language communication. We focus on the symmetric collaborative dialogue setting, where two agents, each with private knowledge, must communicate to achieve a common goal.

This branch contains code for the paper "Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings".

## Dependencies
The following Python packages are required.
- General: Python 2.7, NumPy 1.11, Tensorflow r0.12
- Lexicon: fuzzywuzzy, editdistance
- Web server: TODO
You can also use the docker image `hhexiy/cocoa:0.2`.

## Data collection
### Scenario generation
Generate the schema:
```
mkdir data/cache
python src/scripts/generate_schema.py --schema-path data/schema.json --cache-path data/cache
```
Generate scenarios from the schema:
TODO

### Set up the web server
TODO

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
PYTHONPATH=. python src/main.py --schema-path data/schema.json --scenarios-path data/scenarios.json --train-examples-paths data/train.json --test-examples-paths data/dev.json --stop-words data/common_words.txt --min-epochs 10 --checkpoint checkpoint --rnn-type lstm --learning-rate 0.5 --optimizer adagrad --print-every 50 --model attn-copy-encdec --gpu 1 --rnn-size 100 --grad-clip 0 --num-items 12 --batch-size 32 --stats-file stats.json --entity-encoding-form type --entity-decoding-form type --node-embed-in-rnn-inputs --msg-aggregation max --word-embed-size 100 --node-embed-size 50 --entity-hist-len -1 --learned-utterance-decay
```

## Evaluation
### Test set loss and response generation
Compute the cross entropy and responses generated given ground truth dialogue history:
```
PYTHONPATH=. python src/main.py --schema-path data/schema.json --scenarios-path data/scenarios.json --stop-words data/common_words.txt --init-from checkpoint --test-examples-paths data/test.json --test --batch-size 32 --best --stats-file stats.json --decoding sample 0 --verbose
```

### Bot-bot chat
Dialogue modesl are launched by a system in one ore more sessions (see `src/basic/systems` and `src/basic/sessions`).
Generate bot-bot chat:
```
PYTHONPATH=. python src/scripts/generate_dataset.py --max-turns 46 --schema-path data/schema.json --scenarios-path data/scenarios.json --stop-words data/common_words.txt --test-examples-paths chat.json --train-max-examples 0 --agents neural neural --model-path model/checkpoint --decoding sample 0.5 select
```
To use the rule-based bot, set `--agents simple simple`.

### Analysis
To compute various statistics given a chat transcript, run
```
PYTHONPATH=. python src/scripts/get_data_statistics.py --transcripts chat.json --schema-path data/schema.json --scenarios-path data/scenarios.json --stop-words data/common_words.txt --stats-output dialogue_stats.json
```
The results are saved in `dialogue_stats.json`.

### Visualization
There are two ways to visualize a chat transcript (in json).
To output a plain HTML file that contains all chats, run
```
PYTHONPATH=. python src/scripts/visualize_data.py --transcripts chat.json --schema-path data/schema.json --scenarios-path data/scenarios.json --html-output chat.html
```
To display metadata of each chat (e.g., outcome, number of attributes) and perform filter/search, run
```
PYTHONPATH=. python src/scripts/visualize_data.py --transcripts chat.json --schema-path data/schema.json --scenarios-path data/scenarios.json --html-output chat.html --html-output . --viewer-mode
```
Each chat is then saved as a separate HTML files in `chat_htmls`. To view the chats, open `chat_viewer/chat.html` in a browser.










