model=encdec
exp=basic
lr=0.5
checkpoint=/scr/hehe/game-dialogue/checkpoint/$(exp)
mappings=/scr/hehe/game-dialogue/mappings
stats_file=/scr/hehe/game-dialogue/checkpoint/$(exp).stats
glove=/scr/hehe/game-dialogue/glove/glove.840B.300d.txt
print=50
gpu=1
rnn_size=50
seed=1
max_epochs=100
min_epochs=50
batch_size=128
data=negotiation

dataset:
	PYTHONPATH=. python src/scripts/split_dataset.py --example-paths web_output/combined/transcripts/transcripts.json --train-frac 0.8 --test-frac 0.1 --dev-frac 0.1 --output-path data/$(data)/

train-neg:
	PYTHONPATH=. python src/main.py --schema-path data/$(data)/craigslist-schema.json --train-examples-paths data/$(data)/train.json --test-examples-paths data/$(data)/dev.json --checkpoint $(checkpoint) --min-epochs $(min_epochs) --max-epochs $(max_epochs) --mappings $(mappings) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 5 --batch-size $(batch_size) --stats-file $(stats_file).train --word-embed-size 300 --decoder rnn --dropout 0.5 --context category --context-size 10 --context-encoder bow --decoding sample 0.2 --pretrained-wordvec $(glove) #--train-max-examples 50 --test-max-examples 50 --verbose

test-neg:
	PYTHONPATH=. python src/main.py --schema-path data/$(data)/craigslist-schema.json --test-examples-paths data/$(data)/dev.json --train-examples-paths data/$(data)/train.json --print-every $(print) --gpu $(gpu) --batch-size $(batch_size) --test --stats-file $(stats_file).test --verbose --mappings $(mappings) --best --model encdec --init-from $(checkpoint) #--test-max-examples 4 

index:
	PYTHONPATH=. python src/model/negotiation/retriever.py --schema-path data/$(data)/craigslist-schema.json --train-examples-paths data/$(data)/train.json --index /scr/hehe/game-dialogue/index --retriever-context-len 2 --rewrite-index 

split=dev
retrieve:
	PYTHONPATH=. python src/model/negotiation/retriever.py --schema-path data/$(data)/craigslist-schema.json --test-examples-paths data/$(data)/$(split).json --index /scr/hehe/game-dialogue/index --retriever-context-len 2 --retriever-output /scr/hehe/game-dialogue/$(split)_candidates.json --num-candidates 20 #--test-max-examples 4
