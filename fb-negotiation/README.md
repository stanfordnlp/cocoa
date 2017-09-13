**Schema**: `data/craigslist-schema.json`

**Dataset**: on [Codalab](https://codalab.stanford.edu/bundles/0xd37b585db49243adbba3afe3960b42a2/).
Download the `.json` files and put them in `data`.

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
