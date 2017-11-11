**Schema**: `data/movie-schema.json`

**Dataset**: currently requires access to Stanford NLP servers, please ask for permission
# /juicier/scr105/scr/derekchen14/movie_data/rotten/all.json
# /juicy/scr61/scr/nlp/hehe/cocoa/open-movie/data/handcoded_templates.csv

All command below must be run in the task directory, i.e. `open-movies`.

Let populate the KB to generate the lexicon that finds movies in dialog:
```
PYTHONPATH=. python core/lexicon.py --movie-data data/all.json --output data/lexicon.pkl
```

To run the rulebased bot against itself,
```
PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/movie-schema.json \
--scenarios-path data/toy-scenarios.json \
--train-examples-paths data/rulebased-transcripts.json --train-max-examples 1 \
--test-max-examples 0  --max-turns 20 \
--agents rulebased rulebased
```
Path to output transcripts is given by `--train-examples-paths`.
