scenario:
	python src/basic/generate_scenarios_consistent.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items 5 --num-scenarios 10

dataset:
	python src/basic/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 100 --test-max-examples 100

train-encdec:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --max-epochs 10 --checkpoint checkpoint --rnn-type lstm --learning-rate 1 --optimizer adagrad --print-every 50

train-attnencdec:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --max-epochs 10 --checkpoint checkpoint --rnn-type lstm --learning-rate 1 --optimizer adagrad --print-every 50 --model attn-encdec --grad-clip 5

test:
	python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --max-epochs 1 --checkpoint checkpoint --init-from checkpoint --rnn-type lstm --test-examples-paths output/friends-test-examples.json --test --verbose
