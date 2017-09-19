PYTHONPATH=. python ../scripts/generate_dataset.py \
    --schema-path data/bookhatball-schema.json \
    --scenarios-path data/toy-scenarios.json \
    --train-examples-paths data/toy-transcripts.json \
    --train-max-examples 5 --test-max-examples 0 --max-turns 20 \
    --agents rulebased rulebased\
    --verbose --random-seed 3

# PYTHONPATH=. python ../scripts/generate_dataset.py \
#     --schema-path data/craigslist-schema.json \
#     --scenarios-path data/test-scenarios.json \
#     --train-examples-paths data/test-transcripts.json \
#     --train-max-examples 1 --test-max-examples 0 --max-turns 20 \
#     --agents rulebased cmd --verbose \
#     --price-tracker-model price-tracker.pkl \
#     --random-seed 8
