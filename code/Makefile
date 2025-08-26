setup:
	pip install -r requirements.txt | cat

train:
	python train_predictor.py --config configs/base.yaml | cat

rl:
	python train_rl.py --config configs/base.yaml | cat

eval:
	python eval.py --config configs/base.yaml --weights models/gnn_lstm_model.pth | cat

prepare:
	python prepare_data.py --config configs/base.yaml | cat

e2e:
	$(MAKE) prepare && $(MAKE) train && $(MAKE) rl && $(MAKE) eval

demo:
	python prepare_data.py --config configs/toy.yaml | cat && \
	python train_predictor.py --config configs/toy.yaml --out models/gnn_lstm_model.pth | cat && \
	python train_rl.py --config configs/toy.yaml --out models/rl_agent_q_network.pth --use_predictor_env | cat && \
	python eval.py --config configs/toy.yaml --weights models/gnn_lstm_model.pth | cat

test:
	pytest -q | cat


