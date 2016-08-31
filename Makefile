model=encdec
lr=1
checkpoint=/scr/hehe/game-dialogue/checkpoint/$(model)
print=50
gpu=0

scenario:
	python src/basic/generate_scenarios_consistent.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items 5 --num-scenarios 10

dataset:
	python src/basic/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 100 --test-max-examples 100

train:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --max-epochs 10 --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu)

test:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --init-from $(checkpoint) --rnn-type lstm --test-examples-paths output/friends-train-examples.json --test --verbose --model $(model) --gpu $(gpu)

