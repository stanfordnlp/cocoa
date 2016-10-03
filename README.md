Initialize:

    mkdir output

Generate schema:

    mkdir data/cache
    python src/scripts/generate_schema.py --schema-path data/friends-schema-large.json --cache-path data/cache

Generate scenarios from the schema:

    python src/scripts/generate_scenarios.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items 5 --num-scenarios 10

Generate dataset of dialogues:

    python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 10

Train a model:

    TODO
