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
