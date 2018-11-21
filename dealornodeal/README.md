Code in this folder contains implementation for the DealOrNoDeal task in the following paper:

[Decoupling Strategy and Generation in Negotiation Dialogues](https://arxiv.org/abs/1808.09637).
He He, Derek Chen, Anusha Balakrishnan and Percy Liang.
Empirical Methods in Natural Language Processing (EMNLP), 2018.

Documentation below largely follows steps for [CraigslistBargin](../craigslistbargain/README.md).

## Dependencies
Python 2.7, PyTorch 0.4.

Install `cocoa`:
```
cd ..;
python setup.py develop;
```

## Dataset
**Schema**: `data/bookhatball-schema.json`.

We have converted the Facebook [data](https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate) into our [JSON format](../README.md#examples-and-datasets):
`data/{train,val,test}.json`.

## Building the bot

### Use the modular approach
The modular framework consists of three parts: the parser, the manager, and the generator.

#### 1. Parse the training dialogues.
Parse both training and validation data.
```
PYTHONPATH=. python parse_dialogue.py --transcripts data/train.json --max-examples -1 --templates-output templates.pkl --model-output model.pkl --transcripts-output data/train-parsed.json
PYTHONPATH=. python parse_dialogue.py --transcripts data/val.json --max-examples -1 --templates-output templates.pkl --model-output model.pkl --transcripts-output data/val-parsed.json
```

#### 2. Learning the manager.
We train a seq2seq model over the coarse dialogue acts using parsed data.
```
mkdir -p mappings/lf2lf;
mkdir -p cache/lf2lf;
mkdir -p checkpoint/lf2lf;
PYTHONPATH=. python main.py --schema-path data/bookhatball-schema.json --train-examples-paths data/train-parsed.json --test-examples-paths data/val-parsed.json \
--model lf2lf \
--model-path checkpoint/lf2lf --mappings mappings/lf2lf \
--word-vec-size 200 --pretrained-wordvec '' '' --kb-embed-size 200 \
--rnn-size 200 --rnn-type LSTM --global-attention multibank_general \
--enc-layers 2 --dec-layers 2 \
--num-context 2 \
--batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01 \
--epochs 15 --report-every 500 \
--cache cache/lf2lf \
--verbose --ignore-cache
```

#### <a name=rl>3. Finetune the manager with reinforcement learning.</a>
Generate self-play dialogues using the above learned policy and
run REINFORCE with a given reward function.

First, let's generate the training and validation scenarios.
We will directly get those from the training and validation data.
```
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/train.json --scenarios data/train-scenarios.json
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/val.json --scenarios data/val-scenarios.json
```
Now, we can run self-play and REINFORCE with a reward function, e.g. `margin`.
```
mkdir checkpoint/lf2lf-margin;
PYTHONPATH=. python reinforce.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--agent-checkpoints checkpoint/lf2lf/model_best.pt checkpoint/lf2lf/model_best.pt \
--model-path checkpoint/lf2lf-margin \
--optim adagrad --learning-rate 0.001 \
--agents pt-neural pt-neural \
--report-every 500 --max-turns 20 --num-dialogues 5000 \
--sample --temperature 0.5 --max-length 20 --reward margin
```
- `--reward`: `margin` (utility), `fair` (fairness), and `length` (length).
- `--agents`: agent types 
