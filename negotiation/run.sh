# PYTHONPATH=. python ../scripts/generate_dataset.py \
#     --schema-path data/craigslist-schema.json \
#     --scenarios-path data/test-scenarios.json \
#     --train-examples-paths data/test-transcripts.json \
#     --train-max-examples 1 --test-max-examples 0 --max-turns 20 \
#     --agents rulebased cmd --verbose \
#     --price-tracker-model price-tracker.pkl \
#     --random-seed 8

# PYTHONPATH=. python web/chat_app.py --port 5000 \
#     --schema-path data/craigslist-schema.json --templates templates.pkl \
#     --config web/app_params.json --scenarios-path data/sample-scenarios.json \
#     --output web/output --price-tracker price-tracker.pkl\

# PYTHONPATH=. python ../scripts/visualize_transcripts.py \
#     --dialogue-transcripts web/output/transcripts/transcripts.json \
#     --survey-transcripts web/output/transcripts/surveys.json \
#     --html-output web/output/transcripts/transcripts.html \
#     --img-path images --css-file ../chat_viewer/css/my.css

PYTHONPATH=. python pt_main.py --schema-path data/craigslist-schema.json \
      --train-examples-paths data/train.json --test-examples-paths data/dev.json \
      --train-max-examples 250 --test-max-examples 250 --verbose \
      --stats-file data/stats.train --mappings data/mappings \
      --pretrained-wordvec data/mappings/glove.pt --word-vec-size 300

# PYTHONPATH=. python ../scripts/split_dataset.py --example-paths scr/web_output/combined/transcripts/transcripts.json --train-frac 0.8 --test-frac 0.1 --dev-frac 0.1 --output-path data/

# PYTHONPATH=. python ../scripts/split_dataset.py --example-paths /juicy/scr61/scr/nlp/hehe/cocoa/open-movie/web_output/combined/transcripts/transcripts.json --train-frac 0.8 --test-frac 0.1 --dev-frac 0.1 --output-path data/

# PYTHONPATH=. python ../cocoa/neural/embeddings_to_torch.py \
#     --emb-file /scr/derekchen14/datasets/glove/glove.840B.300d.txt \
#     --vocab-file data/mappings/vocab.pkl --output-file data/mappings/glove --verbose
