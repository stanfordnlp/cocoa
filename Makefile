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
batch_size=10

scenario:
	PYTHONPATH=src python src/scripts/generate_scenarios.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items $(num_items) --num-scenarios 500 --random-seed $(seed)

dataset:
	PYTHONPATH=src python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 100 --test-max-examples 100 --agents heuristic heuristic

train:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --max-epochs $(max_epochs) --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 0 --train-max-examples 10 --test-max-examples 10 --num-items $(num_items) --batch-size $(batch_size) --stats-file $(stats_file).train

train-real:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema-large.json --scenarios-path output/friends-scenarios-large.json --train-examples-paths data/train.json --test-examples-paths data/test.json --max-epochs $(max_epochs) --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 0 --num-items $(num_items) --batch-size $(batch_size) --domain MutualFriends --stats-file $(stats_file).train --verbose --train-max-examples 10 --test-max-examples 10

test-real:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema-large.json --scenarios-path output/friends-scenarios-large.json --test-examples-paths data/test.json --init-from $(checkpoint) --print-every $(print) --model $(model) --gpu $(gpu) --batch-size $(batch_size) --domain MutualFriends --test --best --stats-file $(stats_file).test --verbose --test-max-examples 10

test:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --init-from $(checkpoint) --rnn-type lstm --test-examples-paths output/friends-test-examples.json --test --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --test-max-examples 10 --num-items $(num_items) --batch-size $(batch_size) --best --stats-file $(stats_file).test

bot-chat:
	PYTHONPATH=. python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --test-examples-paths output/friends-bot-chat-$(model).json --train-max-examples 0 --test-max-examples 5 --agents neural neural --model-path $(checkpoint)  

app:
	PYTHONPATH=. python src/web/start_app.py --port 5000 --schema-path data/friends-schema.json --config data/web/app_params.json --scenarios-path output/friends-scenarios.json
