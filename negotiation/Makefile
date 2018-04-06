model=encdec
exp=decay
lr=0.031
scr=/scr/hehe/game-dialogue
glove=/scr/hehe/glove/glove.840B.300d.txt
mappings=scr/mappings-10k
checkpoint=scr/checkpoint/$(exp)
summary_dir=$(checkpoint)/summary
stats_file=$(checkpoint)/stats
cache=basic
cache_path=/u/nlp/scr/hehe/game-dialogue/data_cache/$(cache)
print_every=400
gpu=1
rnn_size=300
num_items=5
seed=1
max_epochs=100
min_epochs=10
num_epochs=50
batch_size=128
entity_form=type
data=negotiation
port=5040
cat=furniture
# cat = category
ckpt_file=$(checkpoint)/model_loss3.02_e13.pt
# path to checkpoint of a trained model

tensorboard:
	fuser 6006/tcp -k; \
	tensorboard --logdir=$(summary_dir);

slot_scores=/scr/hehe/game-dialogue/slot_scores.json

test-scenarios:
	PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/test.json --scenarios data/test-scenarios.json

scenario:
	PYTHONPATH=. python src/scripts/generate_scenarios.py --schema-path data/friends-random-large/schema.json --scenarios-path output/friends-scenarios.json --num-items $(num_items) --num-scenarios 1 --random-seed $(seed)

scrape-craigslist:
	if [ -f scraper/data/negotiation/craigslist_$(cat).json ]; then \
		rm scraper/data/negotiation/craigslist_$(cat).json; \
	fi;
	cd scraper; scrapy crawl craigslist -o data/negotiation/craigslist_$(cat).json -a cache_dir=/scr/hehe/game-dialogue/craigslist_cache -a from_cache=False -a num_result_pages=100 -a category=$(cat) -a image=1

craigslist-scenario:
	PYTHONPATH=. python src/scripts/negotiation/generate_scenarios.py --num-scenarios 3 --schema-path data/negotiation/craigslist-schema.json --scenarios-path data/negotiation/craigslist-scenarios.json --scraped-data scraper/data/negotiation/craigslist_car.json scraper/data/negotiation/craigslist_furniture.json scraper/data/negotiation/craigslist_housing.json --fractions 1 1 1 --intersection 0.2 --flexibility 0.3

dataset:
	PYTHONPATH=. python src/scripts/generate_dataset.py --schema-path data/friends-random-large/schema.json --scenarios-path data/friends-random-large/scenarios.json --train-examples-paths /tmp/$(data)-train-examples.json --test-examples-paths /tmp/$(data)-test-examples.json --train-max-examples 1 --test-max-examples 0 --agents heuristic heuristic --remove-fail --joint-facts

real-dataset:
	PYTHONPATH=. python src/scripts/split_dataset.py --example-paths web_output/combined/transcripts/transcripts.json --train-frac 0.8 --test-frac 0.1 --dev-frac 0.1 --output-path data/$(data)/

glove-pt:
	PYTHONPATH=. python ../cocoa/neural/embeddings_to_torch.py --emb-file $(glove) --vocab-file $(mappings)/vocab.pkl --output-file scr/glove

train:
	PYTHONPATH=. python pt_main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --test-examples-paths data/dev.json \
	--mappings $(mappings) --stats-file $(stats_file).train --pretrained-wordvec scr/glove.pt --word-vec-size 300 --model-path $(checkpoint) \
	--price-tracker $(price_tracker) --batch-size $(batch_size) --gpuid 0 --rnn-size $(rnn_size) --optim adagrad --learning-rate 0.01 --rnn-type LSTM \
	--report-every 500 --epochs $(num_epochs) --verbose  # --train-max-examples 250 --test-max-examples 250 --ignore-cache

test:
	PYTHONPATH=. python evaluate.py --schema-path data/craigslist-schema.json --test-examples-paths data/dev.json --price-tracker $(price_tracker) --batch-size 128 --gpuid 0 --verbose --checkpoint-file $(ckpt_file) \
	--test-max-examples 10 \
	--ignore-cache \


dropout=0.0
train-neg:
	PYTHONPATH=. python pt_main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --test-examples-paths data/dev.json \
	--mappings $(mappings) --stats-file $(stats_file).train --pretrained-wordvec scr/glove.pt --word-vec-size 300 --model-path $(checkpoint) \
	--price-tracker $(price_tracker) --batch-size $(batch_size) --gpuid 0 --rnn-size $(rnn_size) --optim adagrad --learning-rate $(lr) --rnn-type LSTM \
	--report-every $(print_every) --epochs 50  --context title  # --train-max-examples 250 --test-max-examples 250 --ignore-cache

	# --checkpoint $(checkpoint) --min-epochs $(min_epochs) --max-epochs $(max_epochs) --grad-clip 5  --decoder rnn-attn --cache $(cache_path)
	# --context-size 300 --context-encoder bow --num-context 2 --dropout $(dropout) --attention-memory title description --num-candidates 20 \
	# --candidates-path /scr/hehe/game-dialogue/train_candidates.json /scr/hehe/game-dialogue/dev_candidates.json --index /scr/hehe/game-dialogue/index \
	#--test-max-examples 10 --train-max-example 100 #--verbose #--summary-dir $(summary_dir) #--predict-price #--test-max-examples 10 --train-max-example 100
	#--slot-filling --slot-scores $(slot_scores) --dropout 0.5 #--retrieve --index /scr/hehe/game-dialogue/index --rewrite-index --retriever-context-len 2
	#--num-candidates 20 --candidates-path /scr/hehe/game-dialogue/dev_candidates.json /scr/hehe/game-dialogue/train_candidates.json
	#--sample-targets #--sampled-loss #--entity-decoding-form type --entity-target-form type

train-neg-ir:
	PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --test-examples-paths data/dev.json --checkpoint $(checkpoint) --min-epochs $(min_epochs) --max-epochs $(max_epochs) --mappings $(mappings) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 5 --batch-size $(batch_size) --stats-file $(stats_file).train --price-tracker $(price_tracker) --word-embed-size 300 --decoder rnn-attn --cache $(cache_path) --context category --context-size 300 --context-encoder bow --num-context 2 --dropout $(dropout) --attention-memory title description --num-candidates 20 --candidates-path /scr/hehe/game-dialogue/train_candidates.json /scr/hehe/game-dialogue/dev_candidates.json --index /scr/hehe/game-dialogue/index --pretrained-wordvec $(glove) --ignore-cache #--test-max-examples 10 --train-max-example 100 #--verbose #--summary-dir $(summary_dir) #--predict-price #--test-max-examples 10 --train-max-example 100 --dropout 0.5 --retrieve --index /scr/hehe/game-dialogue/index--ignore-cache --rewrite-index --retriever-context-len 2 --num-candidates 20 --candidates-path /scr/hehe/game-dialogue/dev_candidates.json /scr/hehe/game-dialogue/train_candidates.json #--sample-targets #--sampled-loss #--entity-decoding-form type --entity-target-form type 

index:
	PYTHONPATH=. python model/retriever.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --index scr/index --retriever-context-len 2 --rewrite-index --price-tracker-model $(price_tracker) #--slot-scores $(slot_scores)

price_tracker=/scr/hehe/game-dialogue/price_tracker.pkl
price-tracker:
	PYTHONPATH=. python src/basic/negotiation/price_tracker.py --train-examples-path data/$(data)/train.json --output $(price_tracker)

split=dev
n=16
split-retrieve:
	PYTHONPATH=. python src/scripts/negotiation/split_merge_json.py --input data/$(data)/$(split).json --output $(scr)/retrieval/$(split) -n $(n) --split;
	for i in $$(seq 0 15); do \
		PYTHONPATH=. python src/model/negotiation/retriever.py --schema-path data/$(data)/craigslist-schema.json --test-examples-paths $(scr)/retrieval/$(split).$$i --index /scr/hehe/game-dialogue/index --retriever-context-len 2 --retriever-output $(scr)/retrieval/$(split)_candidates.$$i --num-candidates 20 --price-tracker-model /scr/hehe/game-dialogue/price_tracker.pkl --seed $$i &>$(scr)/retrieval/$(split).$$i.log &\
	done
merge-retrieve:
	PYTHONPATH=. python src/scripts/negotiation/split_merge_json.py --input $(scr)/retrieval/$(split)_candidates --output $(scr)/$(split)_candidates.json -n $(n) --merge;

retrieve:
	PYTHONPATH=. python src/model/negotiation/retriever.py --schema-path data/$(data)/craigslist-schema.json --index /scr/hehe/game-dialogue/index --retriever-context-len 2 --num-candidates 20 --test-examples-paths data/$(data)/train.json --retriever-output $(scr)/tmp_candidates.json --price-tracker $(price_tracker) --verbose --test-max-examples 1 

analyze:
	#PYTHONPATH=. python scripts/analyze.py --transcripts data/train.json data/dev.json data/test.json --price-tracker $(price_tracker) --policy scr/model.pkl --max -1
	PYTHONPATH=. python scripts/analyze.py --transcripts web_output/2017-12-08-test/transcripts/transcripts.json --price-tracker $(price_tracker) --policy scr/model.pkl --max -1 --agent rulebased 

eval-neg:
	PYTHONPATH=. python src/main.py --test --eval --eval-examples-paths web_output/eval/eval_120_contexts/results/eval_results2.json --print-every $(print) --gpu $(gpu) --batch-size $(batch_size) --stats-file $(stats_file).eval --best --init-from $(checkpoint) --decoding sample 0.5 --ignore-cache --verbose #--ranker encdec --temperature 0.5 #--verbose #--cache /u/nlp/scr/hehe/game-dialogue/data_cache --retriever-context-len 2 --num-candidates 20 --ranker ir --candidates-path /scr/hehe/game-dialogue/dev_candidates.json --retrieve --index /scr/hehe/game-dialogue/index 

test-neg:
	python main.py --test --test-examples-paths data/dev.json --print-every $(print) --gpu $(gpu) --batch-size $(batch_size) --stats-file $(stats_file).test --best --init-from $(checkpoint) --decoding sample 0.2 --verbose --eval-output $(exp).json --test-max-examples 1 --model $(model) --eval-modes generation --cache $(cache_path) --mappings $(mappings) --ignore-cache #--candidates-path /scr/hehe/game-dialogue/dev_candidates.json #--retriever-context-len 2 --num-candidates 20 --ranker ir --retrieve --index /scr/hehe/game-dialogue/index --schema-path data/craigslist-schema.json --price-tracker-model $(price_tracker)  

ranker=ir
test-neg-ir:
	PYTHONPATH=. python main.py --test --best --test-examples-paths data/dev.json --gpu $(gpu) --batch-size $(batch_size) --stats-file $(stats_file).test --mappings $(mappings) --verbose --ignore-cache --cache $(cache_path) --candidates-path /scr/hehe/game-dialogue/dev_candidates.json --retrieve --index /scr/hehe/game-dialogue/index --init-from $(checkpoint) --model $(model) --test-max-examples 1 --eval-modes generation --ignore-cache #--slot-fillers $(slot_fillers) 

slot:
	PYTHONPATH=. python src/basic/negotiation/slot_detector.py --schema-path data/$(data)/craigslist-schema.json --transcripts data/$(data)/train.json --price-tracker $(price_tracker) --output $(slot_scores) #--max-examples 20
test-slot:
	PYTHONPATH=. python src/basic/negotiation/slot_detector.py --schema-path data/$(data)/craigslist-schema.json --transcripts data/$(data)/train.json --price-tracker $(price_tracker) --slot-scores $(slot_scores) --max-examples 20 --test

vis:
	#PYTHONPATH=. python src/scripts/visualize_data.py --transcripts data/$(data)/test.json --schema-path data/$(data)/schema.json --scenarios-path data/$(data)/scenarios.json --html-output ~/www/chat.html --css-file chat_viewer/css/my.css
	#PYTHONPATH=. python src/scripts/visualizer.py --dialogue-transcripts output/bot-chat-$(model)-$(exp).json --html-output output/bot-chat.html
	PYTHONPATH=. python src/scripts/visualize_data.py --transcripts output/bot-chat-$(model)-$(exp).json --schema-path data/$(data)/craigslist-schema.json --scenarios-path data/$(data)/craigslist-scenarios.json --html-output ~/www/chat.html --css-file chat_viewer/css/my.css

agent_id=0
config-search:
	PYTHONPATH=. python scripts/simulate_rulebased.py --schema-path data/craigslist-schema.json --scenarios-path data/craigslist-scenarios-1000.json --price-tracker $(price_tracker) --max-turns 20 --agent-id $(agent_id)

rule_model=scr/model.pkl
bot-chat:
	PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/craigslist-schema.json --scenarios-path data/sample-scenarios.json --train-examples-paths data/train.json --train-max-examples 1 --test-max-examples 0 --agents pt-neural cmd --price-tracker $(price_tracker) --mappings $(mappings) --checkpoint $(checkpoint) --checkpoint-file $(ckpt_file) --max-turns 20 --random-seed $(seed) --verbose

	# PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/craigslist-schema.json --scenarios-path data/test-scenarios.json --train-examples-paths data/rulebased_transcripts.json --train-max-examples 1 --test-max-examples 0 --agents rulebased cmd --price-tracker $(price_tracker) --max-turns 20 --templates $(tmplt) --policy $(rule_model) --random-seed $(seed) --verbose 
	# PYTHONPATH=. python ../scripts/generate_dataset.py --schema-path data/craigslist-schema.json --scenarios-path data/test-scenarios.json --train-examples-paths data/neural_transcripts.json --train-max-examples 1 --test-max-examples 0 --agents neural-gen cmd --max-turns 20 --checkpoint $(checkpoint) --mappings $(mappings) --price-tracker $(price_tracker) --decoding sample 0 --retrieve --index scr/index --retriever-context-len 2 --num-candidates 10

dump:
	#PYTHONPATH=. python ../scripts/web/dump_db.py --db web_output/$(outdir)/chat_state.db --output web_output/$(outdir)/transcripts/transcripts.json --surveys web_output/$(outdir)/transcripts/surveys.json --schema data/craigslist-schema.json --scenarios-path data/test-scenarios.json #--batch-results web_output/$(outdir)/batch_results.csv
	PYTHONPATH=. python ../scripts/visualize_transcripts.py --dialogue-transcripts web_output/$(outdir)/transcripts/transcripts.json --survey-transcripts web_output/$(outdir)/transcripts/surveys.json --html-output web_output/$(outdir)/transcripts/transcripts.html --img-path images --css-file ../chat_viewer/css/my.css --summary #--worker-ids web_output/$(outdir)/worker_ids.json

analyze-neg:
	PYTHONPATH=. python src/analysis/negotiation/analyze_strategy.py --output-dir web_output/$(outdir) --price-tracker $(price_tracker) #--html-output web_output/$(outdir)/transcripts/analysis.html --img-path images --html-visualize --mpld3-plugin src/analysis/negotiation/mpld3_plugin.js #--max-examples 2

tmplt=scr/templates.pkl
templates:
	PYTHONPATH=. python systems/templates.py --transcripts data/train.json --price-tracker $(price_tracker) --output $(tmplt) --max-examples -1 --templates $(tmplt)
	#PYTHONPATH=. python ../scripts/visualize_transcripts.py --dialogue-transcripts scr/data/parsed_transcripts.json --survey-transcripts ../web_output/combined/transcripts/surveys.json --html-output scr/data/parsed_transcripts.html --img-path images --css-file ../chat_viewer/css/my.css #--summary #--worker-ids web_output/$(outdir)/worker_ids.json

parse:
	#PYTHONPATH=. python model/parser.py --transcripts data/train.json --price-tracker $(price_tracker) --max-examples -1
	PYTHONPATH=. python parse_dialogue.py --transcripts data/train.json --price-tracker $(price_tracker) --max-examples -1 --templates-output scr/templates.pkl --model-output scr/model.pkl --model scr/model.pkl

outdir=tmp
config=app_params.json
app:
	PYTHONPATH=. python web/chat_app.py --port $(port) --schema-path data/craigslist-schema.json --config web/$(config) --scenarios-path data/test-scenarios.json --num-scenarios 200 --output web_output/$(outdir) --price-tracker $(price_tracker) --templates $(tmplt) --policy $(rule_model) --checkpoint $(checkpoint) --mappings $(mappings) --decoding sample 0.2
	#PYTHONPATH=. python src/web/start_app.py --port $(port) --schema-path data/friends-schema.json --config data/web/app_params.json --scenarios-path output/friends-scenarios.json

analyze-trial:
	PYTHONPATH=. python scripts/analyze_trials.py --trial-db data/20170921-trials.pkl --transcripts web_output/$(outdir)/transcripts/transcripts.json

plot-neg:
	PYTHONPATH=. python src/analysis/negotiation/analyze_strategy.py --output-dir web_output/combined --price-tracker $(price_tracker) --max-examples 100

db=test.json
turk-review:
	PYTHONPATH=. python ../scripts/turk/eval_dialogue.py --aws-config ../data/turk/aws_config.json --question-template ../cocoa/turk/templates/compare_question.html --overall-template ../cocoa/turk/templates/frame.html --instructions turk/templates/compare_instructions.html --hit-db scr/turk/eval_results/negotiation/$(db) --review

turk-eval:
	PYTHONPATH=. python ../scripts/turk/eval_dialogue.py --debug --aws-config ../data/turk/aws_config.json --question-template ../cocoa/turk/templates/multi_question.html --overall-template ../cocoa/turk/templates/frame.html --instructions turk/templates/multi_instructions.html --eval-data scr/turk/eval_data/negotiation/eval_data.json --num-eval 28 --num-assignments-per-hit 5 --num-questions-per-hit 56 --hit-db scr/turk/eval_results/negotiation/$(db) --systems ir generative --num-overlap-questions 0 --title "Dialogue response evaluation" --reward-per-hit 1.00 --qids scr/turk/eval_data/negotiation/test_qids.json #--system-outputs generative scr/turk/eval_data/negotiation/generative_results.json ir scr/turk/eval_data/negotiation/ir_results.json selector scr/turk/eval_data/negotiation/selector_results.json generic scr/turk/eval_data/negotiation/generic_results.json #--remove-evaluated scr/turk/eval-1000-1.json scr/turk/eval-1040-1-ref.json --script turk/templates/hit_limit.js 
