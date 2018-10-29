**Schema**: `data/bookhatball-schema.json`

All command below must be run in the task directory, i.e. `fb-negotiation`.

**Split tracker**: maintain belief of their offer in how the 3 items should be split among both agents.

Let's create some scenarios from the `data/selfplay.txt`:
```
PYTHONPATH=. python scripts/create_scenarios.py --schema-path data/bookhatball-schema.json --scenario-ints-file data/selfplay.txt --output data/test-scenarios.json
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

To set up the website,
```
PYTHONPATH=. python web/chat_app.py --port 5000 --schema-path data/bookhatball-schema.json \
--config web/app_params.json --scenarios-path data/test-scenarios.json \
--output web_output/<dir-name>
```
Arguments:
- `port`: The port of the web server.
- `config`: Path to the configuration file. See an example at `web/app_params.json`. To add a bot, put its arguments in `models`. 
- `output`: Location of the database, error logs, and transcripts. *NOTE*: currently the path exists, then the whole directory is deleted (even files not related to the website) during initialization unless `--reuse` is specified. **TODO**: only delete relevant files.

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
