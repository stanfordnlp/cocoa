port=5040
outdir=tmp
config=app_params.json
lexicon=scr/lexicon.pkl
tmplt=scr/templates.pkl
mappings=scr/mappings
exp=basic
stats_file=scr/experiments/$(exp)/stats
glove=/scr/hehe/glove/glove.840B.300d.txt

split-dataset:
	PYTHONPATH=. python ../scripts/split_dataset.py --example-paths scr/web_output/combined/transcripts/transcripts.json --train-frac 0.8 --test-frac 0.1 --dev-frac 0.1 --output-path data/

glove-pt:
	PYTHONPATH=. python ../cocoa/neural/embeddings_to_torch.py --emb-file $(glove) --vocab-file $(mappings)/vocab.pkl --output-file scr/glove

train:
	PYTHONPATH=. python train.py --schema-path data/schema.json --train-examples-paths data/train.json --test-examples-paths data/dev.json --train-max-examples 250 --test-max-examples 250 --mappings $(mappings) --stats-file $(stats_file).train --pretrained-wordvec scr/glove.pt --word-vec-size 300

dump:
	PYTHONPATH=. python ../scripts/web/dump_db.py --db web_output/$(outdir)/chat_state.db --output web_output/$(outdir)/transcripts/transcripts.json --surveys web_output/$(outdir)/transcripts/surveys.json --schema data/schema.json --scenarios-path data/scenarios.json #--batch-results web_output/$(outdir)/batch_results.csv
	PYTHONPATH=. python ../scripts/visualize_transcripts.py --dialogue-transcripts web_output/$(outdir)/transcripts/transcripts.json --survey-transcripts web_output/$(outdir)/transcripts/surveys.json --html-output web_output/$(outdir)/transcripts/transcripts.html --img-path images --css-file ../chat_viewer/css/my.css --summary #--worker-ids web_output/$(outdir)/worker_ids.json

#movie_db=scr/data/combined.csv
#movie_db=/juicier/scr105/scr/derekchen14/movie_data/small_merged.json
movie_db=/scr/derekchen14/movie_data/all_v10.json
templates:
	PYTHONPATH=. python model/templates.py --movie-data $(movie_db) --templates scr/data/handcoded_templates.csv --output $(tmplt)

lexicon:
	PYTHONPATH=. python core/lexicon.py --movie-data $(movie_db) --output $(lexicon) --lexicon $(lexicon)

parse:
	PYTHONPATH=. python parse_dialogue.py --transcripts data/transcripts.json --lexicon $(lexicon) --max-examples -1 --templates-output $(tmplt) --model-output scr/model.pkl --reviews $(movie_db) 

bot-chat:
	PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/schema.json --scenarios-path data/scenarios.json --train-examples-paths data/rulebased_transcripts.json --train-max-examples 1 --test-max-examples 0 --agents rulebased cmd --max-turns 20 --templates $(tmplt) --lexicon $(lexicon) --policy scr/model.pkl

app:
	PYTHONPATH=. python web/chat_app.py --port $(port) --schema-path data/schema.json --config web/$(config) --scenarios-path data/scenarios.json --num-scenarios 100 --output web_output/$(outdir) --templates $(tmplt) --lexicon $(lexicon) --policy scr/model.pkl

analyze:
	PYTHONPATH=. python ../scripts/analyze.py --transcripts data/transcripts.json --lexicon $(lexicon) --policy scr/model.pkl --max -1
