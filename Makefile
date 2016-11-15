model=encdec
exp=basic
lr=1
checkpoint=/scr/hehe/game-dialogue/checkpoint/$(model)-$(exp)
stats_file=/scr/hehe/game-dialogue/checkpoint/$(model)-$(exp).stats
print=50
gpu=0
rnn_size=50
num_items=5
seed=1
max_epochs=50
batch_size=32
entity_encoding_form=canonical

scenario:
	PYTHONPATH=. python src/scripts/generate_scenarios.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items $(num_items) --num-scenarios 1500 --random-seed $(seed)

dataset:
	PYTHONPATH=. python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 1000 --test-max-examples 500 --agents heuristic heuristic

real-dataset:
	PYTHONPATH=. python src/scripts/split_dataset.py --example-paths data/mutualfriends/transcripts-1.json data/mutualfriends/transcripts-2.json --train-frac 0.7 --test-frac 0.3 --dev-frac 0 --output-path data/mutualfriends/

train:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --max-epochs $(max_epochs) --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 0 --num-items $(num_items) --batch-size $(batch_size) --stats-file $(stats_file).train #--train-max-examples 100 --test-max-examples 100 

train-real:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema-large.json --scenarios-path output/friends-scenarios-large.json output/friends-scenarios-large-peaky.json output/friends-scenarios-large-peaky-04-002.json --train-examples-paths data/mutualfriends/train.json --test-examples-paths data/mutualfriends/test.json --max-epochs $(max_epochs) --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 0 --num-items $(num_items) --batch-size $(batch_size) --domain MutualFriends --stats-file $(stats_file).train --entity-encoding-form $(entity_encoding_form) #--verbose #--train-max-examples 10 --test-max-examples 10

test-real:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema-large.json --scenarios-path output/friends-scenarios-large.json output/friends-scenarios-large-peaky.json output/friends-scenarios-large-peaky-04-002.json --test-examples-paths data/mutualfriends/test.json --init-from $(checkpoint) --print-every $(print) --model $(model) --gpu $(gpu) --batch-size $(batch_size) --domain MutualFriends --test --best --stats-file $(stats_file).test --verbose #--test-max-examples 10

test:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --init-from $(checkpoint) --rnn-type lstm --test-examples-paths output/friends-test-examples.json --test --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --num-items $(num_items) --batch-size $(batch_size) --best --stats-file $(stats_file).test

bot-chat:
	PYTHONPATH=. python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --test-examples-paths output/friends-bot-chat-$(model)-$(exp).json --train-max-examples 0 --test-max-examples 1 --agents neural neural --model-path $(checkpoint) --scenario-offset 1000

app:
	PYTHONPATH=. python src/web/start_app.py --port 5000 --schema-path data/friends-schema.json --config data/web/app_params.json --scenarios-path output/friends-scenarios.json
