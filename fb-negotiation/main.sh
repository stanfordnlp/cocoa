PYTHONPATH=. python ../scripts/generate_dataset.py \
    --schema-path data/bookhatball-schema.json \
    --scenarios-path data/toy-scenarios.json \
    --train-examples-paths data/toy-transcripts.json \
    --train-max-examples 1 --test-max-examples 0 --max-turns 20 \
    --agents rulebased rulebased \
    --verbose # --random-seed 8