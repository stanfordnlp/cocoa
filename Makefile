model=encdec
exp=$(model)
lr=1
checkpoint=/scr/hehe/game-dialogue/checkpoint/$(exp)
print=50
gpu=0
rnn_size=50
num_items=5
seed=1

scenario:
	PYTHONPATH=src python src/scripts/generate_scenarios.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items $(num_items) --num-scenarios 500 --random-seed $(seed)

dataset:
	PYTHONPATH=src python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 100 --test-max-examples 100 --agents heuristic heuristic

train:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-train-examples.json --max-epochs 50 --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 0 --train-max-examples 100 --test-max-examples 100 --num-items $(num_items) --batch-size 10

test:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --init-from $(checkpoint) --rnn-type lstm --test-examples-paths output/friends-train-examples.json --test --verbose --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --test-max-examples 100 --num-items $(num_items) --batch-size 10

