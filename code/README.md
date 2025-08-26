## Epidemiology Forecasting and Optimal Control (Scripted)

This repo now provides reproducible CLI scripts, configs, tests, and installation guidance. Notebooks remain under `nbks/` for exploration, but all core workflows are automated.

### Quickstart

```bash
git clone <repo>
cd epidemiology-Project/code
python -m venv .venv && .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
```

### PyTorch and PyTorch Geometric install

Do NOT `pip install torch-geometric` blindly. Follow the official instructions and match your Torch/CUDA.

- PyTorch install matrix: `https://pytorch.org/get-started/locally/`
- PyG install guide: `https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html`

Known-good CPU-only combo:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

### Scripts

- `prepare_data.py`: Downloads/prepares minimal dataset and persists scalers (synthetic placeholder).
- `train_predictor.py`: Trains GNN-LSTM predictor.
- `train_rl.py`: Trains DQN agent on a minimal env.
- `eval.py`: Loads predictor and reports output shapes.

Use `configs/base.yaml` to control settings.

```bash
python prepare_data.py --config configs/base.yaml
python train_predictor.py --config configs/base.yaml --out models/gnn_lstm_model.pth
python train_rl.py --config configs/base.yaml --out models/rl_agent_q_network.pth
python eval.py --config configs/base.yaml --weights models/gnn_lstm_model.pth
```

Make targets:

```bash
make setup
make prepare
make train
make rl
make eval
make test
```

### Configs

See `configs/base.yaml` for dataset paths, graph normalization, horizons, reward weights, and seeds.

### Tests

- Graph normalization check
- Predictor forward shape
- Deterministic DQN with fixed seed

Run: `pytest -q`.

### Results

| Model | MAE (placeholder) | RMSE (placeholder) | Notes |
|---|---:|---:|---|
| GNN-LSTM | 0.123 | 0.234 | Synthetic demo |

Learning curves and sample outputs:

![LSTM Loss](assets/lstm-loss.png)
![LSTM Result](assets/lstm-result.png)
![RL Result](assets/RL-result.png)

### Data

For real data, consider:

- JHU CSSE time series (archived): `https://github.com/CSSEGISandData/COVID-19`
- OxCGRT NPIs: `https://github.com/OxCGRT/covid-policy-tracker`

`prepare_data.py` is a placeholder; extend it to download, fuse by node list, build adjacency (contiguity or mobility), and persist scalers.

### Graph construction

We use symmetric normalization \( \hat{A} = D^{-1/2}(A + I)D^{-1/2} \). See `src/data/build_graph.py`.

### Method fidelity

RL currently uses a minimal simulator. To mirror a model-based setup: roll out actions against the frozen predictor as a dynamics model.

### Repo hygiene

Large binaries under `models/` and `gnn-lstm-params/` should be moved to a Release or Git LFS.

### License and citation

MIT licensed (see `LICENSE`). Citation metadata in `CITATION.cff`.


