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

test:
	pytest -q | cat


