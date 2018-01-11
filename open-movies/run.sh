# PYTHONPATH=. python ../scripts/generate_dataset.py \
#     --schema-path data/movie-schema.json --agents rulebased cmd \
#     --scenarios-path data/toy-scenarios.json \
#     --train-examples-paths data/toy-transcripts.json \
#     --train-max-examples 1 --test-max-examples 0 --max-turns 20 \
#     --templates data/templates.pkl --lexicon data/lexicon.pkl \
#     --verbose --random-seed 8

PYTHONPATH=. python web/chat_app.py --port 5040 \
    --schema-path data/movie-schema.json --lexicon data/lexicon.pkl \
    --config web/app_params.json --scenarios-path data/toy-scenarios.json \
    --output web/output --templates data/templates.pkl \
    --num-scenarios 100 --policy data/model.pkl

# PYTHONPATH=. python core/lexicon.py --movie-data data/all_merged.json \
#     --output data/lexicon.pkl
# PYTHONPATH=. python model/templates.py --movie-data data/all_merged.json \
#     --templates /juicy/scr61/scr/nlp/hehe/cocoa/open-movie/data/handcoded_templates.csv \
#     --output data/templates.pkl --source rotten #or kaggle

# PYTHONPATH=. python ../scripts/visualize_transcripts.py \
#     --dialogue-transcripts web/output/transcripts/transcripts.json \
#     --survey-transcripts web/output/transcripts/surveys.json \
#     --html-output web/output/transcripts/transcripts.html \
#     --img-path images --css-file ../chat_viewer/css/my.css --task movies
