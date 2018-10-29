Code in this folder contains implementation for the CraigslistBargain task in the following paper:

[Decoupling Strategy and Generation in Negotiation Dialogues](https://arxiv.org/abs/1808.09637).
He He, Derek Chen, Anusha Balakrishnan and Percy Liang.
Empirical Methods in Natural Language Processing (EMNLP), 2018.

## Dependencies
`pip install -r requirements.txt`

## Dataset
All data is on the Codalab [worksheet](https://worksheets.codalab.org/worksheets/0x453913e76b65495d8b9730d41c7e0a0c/).

If you want to collect your own data, read the following steps.

### Scenario generation
1. Schema: `data/craigslist-schema.json`.
2. Scrape Craigslist posts from different categories:
```
cd scraper;
for cat in car phone bike electronics furniture housing; do \
    scrapy crawl craigslist -o data/negotiation/craigslist_$cat.json -a cache_dir=/tmp/craigslist_cache -a from_cache=False -a num_result_pages=100 -a category=$cat -a image=1; \
done
```
3. Generate scenarios: 
```
PYTHONPATH=. python scripts/generate_scenarios.py --num-scenarios <number> --schema-path data/craigslist-schema.json --scenarios-path data/scenarios.json --scraped-data scraper/data/negotiation --categories furniture housing car phone bike electronics --fractions 1 1 1 1 1 1 --discounts 0.9 0.7 0.5
```
- `--fractions`: fractions to sample from each category.
- `--discounts`: possible targets for the buyer, `discount * listing_price`.

### Set up the website and AMT HITs. 
See [data collection](../README.md#data-collection) in `cocoa` README.


## Building the bot

### Use the modular approach
The modular framework consists of three parts: the parser, the manager, and the generator.

#### 1. Build the price tracker.
The price tracker recognizes price mentions in an utterance.
```
PYTHONPATH=. python core/price_tracker.py --train-examples-path data/train.json --output <path-to-save-price-tracker>
```

#### 2. Parse the training dialogues.
```
PYTHONPATH=. python parse_dialogue.py --transcripts data/train.json --price-tracker <path-to-save-price-tracker> --max-examples -1 --templates-output templates.pkl --model-output model.pkl --transcripts-output data/train-parsed.json
```
- Parse utterances into coarse dialogue acts using the rule-based parser (`--transcripts-output`).
- Learn an n-gram model over the dialogue acts (`--model-output`), which will be used by the **hybrid policy**.
- Extract utterance templates (`--templates-output`) for retrieval-based generator.

#### 3. Learning the manager.
We train a seq2seq model over the coarse dialogue acts using parsed data.
```
model='lf2lf'; \
PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train-parsed.json --test-examples-paths data/dev-parsed.json \
--price-tracker price_tracker.pkl \
--stats-file stats.train --model-path checkpoint \
--model $model \
--mappings mappings/$model --word-vec-size 300 --pretrained-wordvec '' '' \
--batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01 \
--rnn-size 300 --rnn-type LSTM --global-attention multibank_general \
--num-context 2 --stateful \
--epochs 15 \
--report-every 500 \
--cache cache/$model \
--verbose \
```

#### 4. Finetuning the manager with reinforcement learning.
Generate self-play dialogues using the above learned policy and
run REINFORCE with a given reward function.
```
model='lf2lf'; \
PYTHONPATH=. python reinforce.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker price_tracker.pkl \
--mappings mappings/$model \
--agent-checkpoints <ckpt-file> <ckpt-file> \
--model-path rl-checkpoint \
--optim adagrad --learning-rate 0.001 --discount-factor 0.95 \
--agents pt-neural pt-neural \
--report-every 500 --max-turns 20 --num-dialogues 5000 \
--sample --temperature 0.5 --max-length 20 --reward <reward-function>
```

==== DEPRECATED ====


All command below must be run in the task directory, i.e. `negotiation`.

**Price tracker**: detect mentions of prices (entities) in an utterance.

We first "train" a price tracker by weak supervision from the dollar sign ($) in front of prices.
```
PYTHONPATH=. python core/price_tracker.py --train-examples-path data/train.json --output price-tracker.pkl
```

Let's use scenarios from the test set to test the bot:
```
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/test.json --scenarios data/test-scenarios.json
```

To run the rulebased bot against itself,
```
PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/craigslist-schema.json \
--scenarios-path data/test-scenarios.json \
--train-examples-paths data/rulebased-transcripts.json --train-max-examples 1 \
--test-max-examples 0 \
--agents rulebased rulebased --price-tracker price-tracker.pkl --max-turns 20 
```
Path to output transcripts is given by `--train-examples-paths`.

To set up the website,
```
PYTHONPATH=. python web/chat_app.py --port 5000 --schema-path data/craigslist-schema.json \
--config web/app_params.json --scenarios-path data/sample-scenarios.json \
--output web_output/<dir-name> --price-tracker price-tracker.pkl
```
Arguments:
- `port`: The port of the web server.
- `config`: Path to the configuration file. See an example at `web/app_params.json`. To add a bot, put its arguments in `models`. 
- `output`: Location of the database, error logs, and transcripts. *NOTE*: currently the path exists, then the whole directory is deleted (even files not related to the website) during initialization unless `--reuse` is specified. **TODO**: only delete relevatn files.

The chat data is logged in `<output>/chat_state.db`.
Upon exit, the server dumps the data to `<output>/transcripts/transcripts.json` and `<output>/transcripts/survey.json`.

To dump the data explicitly (e.g. while the server is running), run
```
PYTHONPATH=. python ../scripts/web/dump_db.py --db <path_to_chat_state.db> \
--output <path_to_transcripts.json> --surveys <path_to_surveys.json> \
--schema data/craigslist-schema.json --scenarios-path data/sample-scenarios.json 
```

To visualize the data in HTML,
```
PYTHONPATH=. python ../scripts/visualize_transcripts.py \
--dialogue-transcripts <path_to_transcripts.json> --survey-transcripts <path_to_surveys.json> \
--html-output <path_to_transcripts.html> --img-path images \
--css-file ../chat_viewer/css/my.css
```
Arguments:
- `images`: image path used in HTML.
- `summary`: if specified, a summary of survey evaluation scores for different systems will be printed.

Basic chat visualization functions are provided in `cocoa/analysis/visualizer` and `cocoa/analysis/html_visualizer`. To extend those to a new task, you can derive new classes (`Visualizer` and `HTMLVisualizer`) in `task/analysis`.
