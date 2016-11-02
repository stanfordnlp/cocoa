model=encdec
exp=$(model)
lr=1
checkpoint=/scr/hehe/game-dialogue/checkpoint/$(exp)
print=50
gpu=0
rnn_size=50
num_items=5
seed=1
max_epochs=50

scenario:
	PYTHONPATH=src python src/scripts/generate_scenarios.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --num-items $(num_items) --num-scenarios 500 --random-seed $(seed)

dataset:
	PYTHONPATH=src python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --train-max-examples 100 --test-max-examples 100 --agents heuristic heuristic

train:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --train-examples-paths output/friends-train-examples.json --test-examples-paths output/friends-test-examples.json --max-epochs $(max_epochs) --checkpoint $(checkpoint) --rnn-type lstm --learning-rate $(lr) --optimizer adagrad --print-every $(print) --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --grad-clip 0 --train-max-examples 100 --test-max-examples 100 --num-items $(num_items) --batch-size 10

test:
	PYTHONPATH=. python src/main.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --init-from $(checkpoint) --rnn-type lstm --test-examples-paths output/friends-test-examples.json --test --verbose --model $(model) --gpu $(gpu) --rnn-size $(rnn_size) --test-max-examples 10 --num-items $(num_items) --batch-size 10 --best

bot-chat:
	PYTHONPATH=. python src/scripts/generate_dataset.py --schema-path data/friends-schema.json --scenarios-path output/friends-scenarios.json --test-examples-paths output/friends-bot-chat-$(model).json --train-max-examples 0 --test-max-examples 1 --agents neural neural --model-path $(checkpoint)  

app:
	PYTHONPATH=. python src/web/start_app.py --port 5000 --schema-path data/friends-schema.json --config data/web/app_params.json --scenarios-path output/friends-scenarios.json
