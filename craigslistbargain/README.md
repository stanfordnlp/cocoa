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

#### <a name=price-tracker>1. Build the price tracker.</a>
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
mkdir -p mappings/lf2lf;
mkdir -p cache/lf2lf;
mkdir -p checkpoint/lf2lf;
PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train-parsed.json --test-examples-paths data/dev-parsed.json \
--price-tracker price_tracker.pkl \
--model lf2lf \
--model-path checkpoint/lf2lf --mappings mappings/lf2lf \
--word-vec-size 300 --pretrained-wordvec '' '' \
--rnn-size 300 --rnn-type LSTM --global-attention multibank_general \
--num-context 2 --stateful \
--batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01 \
--epochs 15 --report-every 500 \
--cache cache/lf2lf \
--verbose
```

#### 4. Finetuning the manager with reinforcement learning.
Generate self-play dialogues using the above learned policy and
run REINFORCE with a given reward function.

First, let's generate the training and validation scenarios.
We will directly get those from the training and validation data.
```
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/train.json --scenarios data/train-scenarios.json
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/dev.json --scenarios data/dev-scenarios.json
```
Now, we can run self-play and REINFORCE with a reward function, e.g. `margin`.
```
mkdir checkpoint/lf2lf-margin;
PYTHONPATH=. python reinforce.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker price_tracker.pkl \
--mappings mappings/lf2lf \
--agent-checkpoints checkpoint/lf2lf/model_best.pt checkpoint/lf2lf/model_best.pt \
--model-path checkpoint/lf2lf-margin \
--optim adagrad --learning-rate 0.001 \
--agents pt-neural pt-neural \
--report-every 500 --max-turns 20 --num-dialogues 5000 \
--sample --temperature 0.5 --max-length 20 --reward margin
```
- `--reward`: `margin` (utility), `fair` (fairness), and `length` (length).

### Use the end-to-end approach

#### 1. Build pretrained word embeddings.
First, build the vocabulary. Note that we need the [price tracker](#price-tracker) to bin prices.
```
mkdir -p mappings/seq2seq;
PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths scr/data/train.json --mappings mappings/seq2seq --model seq2seq --price-tracker price_tracker.pkl --ignore-cache --vocab-only
```

Get the GloVe embedding.
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip;
unzip glove.840B.300d.zip;
```

Filter pretrained embedding for the model vocab.
We use separate embeddings for the utterances and the product description specified by `--vocab-type`.
```
PYTHONPATH=. python ../cocoa/neural/embeddings_to_torch.py --emb-file glove.840B.300d.txt --vocab-file mappings/seq2seq/vocab.pkl --output-file mappings/seq2seq/ --vocab-type kb
PYTHONPATH=. python ../cocoa/neural/embeddings_to_torch.py --emb-file glove.840B.300d.txt --vocab-file mappings/seq2seq/vocab.pkl --output-file mappings/seq2seq/ --vocab-type utterance
```

#### 2. Train the seq2seq model.
```
mkdir -p cache/seq2seq;
mkdir -p checkpoint/seq2seq;
PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --test-examples-paths data/dev-parsed.json \
--price-tracker price_tracker.pkl \
--model seq2seq \
--model-path checkpoint/seq2seq --mappings mappings/seq2seq \
--pretrained-wordvec mappings/seq2seq/utterance_glove.pt mappings/seq2seq/kb_glove.pt --word-vec-size 300 \
--rnn-size 300 --rnn-type LSTM --global-attention multibank_general \
--enc-layers 2 --dec-layers 2 --num-context 2 \
--batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01  \
--report-every 500 \
--epochs 15 \
--cache cache/seq2seq \
--verbose
```

## Chat with the bot
Chat with the bot in the command line interface:
```
PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/craigslist-schema.json --scenarios-path data/dev-scenarios.json --results-paths bot-chat-transcripts.json --max-examples 20 --agents <agent-name> cmd --price-tracker price_tracker.pkl --agent-checkpoints <ckpt-file> "" --max-turns 20 --random-seed <seed> --sample --temperature 0.2
```
Chat with the bot in the web interface:


===DEPRECATED===

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
