**Schema**: `data/craigslist-schema.json`

**Dataset**: on [Codalab](https://codalab.stanford.edu/bundles/0xd37b585db49243adbba3afe3960b42a2/).
Download the `.json` files and put them in `data`.

All command below must be run in the task directory, i.e. `fb-negotiation`.

**Split tracker**: maintain belief of their offer in how the 3 items should be split among both agents.


Let's use scenarios from the test set to test the bot:
```
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/test.json --scenarios data/test-scenarios.json
```

To run the rulebased bot against itself,
```
PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/bookhatball-schema.json \
--scenarios-path data/toy-scenarios.json \
--train-examples-paths data/rulebased-transcripts.json --train-max-examples 1 \
--test-max-examples 0  --max-turns 20 \
--agents rulebased rulebased
```
Path to output transcripts is given by `--train-examples-paths`.
