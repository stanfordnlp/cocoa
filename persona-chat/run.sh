PYTHONPATH=. python ../scripts/generate_dataset.py \
    --schema-path data/persona-schema.json \
    --scenarios-path data/test_revised_scenarios.json \
    --train-examples-paths data/test_revised_examples.json \
    --train-max-examples 1 --test-max-examples 0 --max-turns 20 \
    --agents rulebased cmd --templates data/templates.pkl \
    --verbose --random-seed 8 --policy data/policy.pkl

# PYTHONPATH=. python web/chat_app.py --port 1414 \
#     --schema-path data/bookhatball-schema.json \
#     --config web/app_params.json --scenarios-path data/test-scenarios.json \
#     --output web/output

# PYTHONPATH=. python ../scripts/visualize_transcripts.py \
#     --dialogue-transcripts web/output/transcripts/transcripts.json \
#     --survey-transcripts web/output/transcripts/surveys.json \
#     --html-output web/output/transcripts/transcripts.html \
#     --img-path images --css-file ../chat_viewer/css/my.css --task fb-neg