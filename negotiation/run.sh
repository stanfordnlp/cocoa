# PYTHONPATH=. python negotiation/web/chat_app.py --port 5000 \
#     --schema-path negotiation/data/craigslist-schema.json \
#     --config negotiation/web/app_params.json --scenarios-path negotiation/data/sample-scenarios.json \
#     --output negotiation/web/output --price-tracker negotiation/price-tracker.pkl \

PYTHONPATH=. python web/chat_app.py --port 5000 \
    --schema-path data/craigslist-schema.json \
    --config web/app_params.json --scenarios-path data/sample-scenarios.json \
    --output web/output --price-tracker price-tracker.pkl \

# PYTHONPATH=. python ../scripts/generate_dataset.py \
#     --schema-path data/craigslist-schema.json \
#     --scenarios-path data/test-scenarios.json \
#     --train-examples-paths data/test-transcripts.json \
#     --train-max-examples 1 --test-max-examples 0 --max-turns 20 \
#     --agents rulebased cmd --verbose \
#     --price-tracker-model price-tracker.pkl \
#     --random-seed 8
