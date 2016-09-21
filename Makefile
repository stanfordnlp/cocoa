model=encdec
exp=$(model)
lr=1
checkpoint=/scr/hehe/game-dialogue/checkpoint/$(exp)
print=50
gpu=0
rnn_size=50
seed=1

scenario:
	python src/basic/generate_scenarios.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items 5 --num-scenarios 500 --random-seed $(seed)

dataset:
	python src/basic/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 100 --test-max-examples 100 --agents heuristic heuristic

train:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-train-examples.json --max-epochs 100 --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --train-utterance --grad-clip 0 --train-max-examples 10 --test-max-examples 10 

test:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --init-from $(checkpoint) --rnn-type lstm --test-examples-paths output/friends-train-examples.json --test --verbose --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --train-utterance --test-max-examples 10

