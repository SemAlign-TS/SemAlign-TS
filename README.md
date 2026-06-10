# SemAlign-TS

Semantic-aligned time series generation with diffusion pretraining and OSRA (Online Semantic Reward Alignment) fine-tuning.

## Project Structure

```
SemAlign-TS/
├── Data/                          # Preprocessed datasets (not included in repo)
│   └── {dataset}/
│       ├── {dataset}_X.npy
│       ├── {dataset}_emb.npy
│       ├── {dataset}_emb_mask.npy
│       ├── {dataset}_meta.pkl
│       └── {dataset}_train_indices.npy
├── diff_train.py                  # Step 1: Diffusion pretraining
├── osra_train.py                  # Step 2: OSRA fine-tuning
├── evaluate.py                    # Step 3: Evaluation
├── diffusion_core/                # Diffusion model and scheduler
├── reward_core/                   # Semantic reward functions
├── checkpoints/                   # Saved models (created at runtime)
├── logs/                          # Training logs (created at runtime)
└── evaluation_results/            # Evaluation outputs (created at runtime)
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)

Third-party Python packages (everything else is stdlib):

| Package | Used for |
|---------|----------|
| `torch` | Model, training, sampling |
| `numpy` | Data I/O, metrics |
| `tqdm` | Evaluation progress bar |

```bash
# GPU: install PyTorch for your CUDA version first (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Data Preparation

Place each preprocessed dataset under `Data/{dataset}/`. For example, ETTh1:

```
Data/etth1/
├── etth1_X.npy
├── etth1_emb.npy
├── etth1_emb_mask.npy
├── etth1_meta.pkl
└── etth1_train_indices.npy
```

## Reproduction Pipeline

Run the three stages in order from the project root:

### Step 1 — Diffusion Pretraining

```bash
python diff_train.py --dataset etth1
```

Checkpoints are saved to `checkpoints/diff_train/etth1/best_model.pt`.

### Step 2 — OSRA Fine-tuning

```bash
python osra_train.py --dataset etth1
```

Loads the diffusion checkpoint from Step 1 and saves the OSRA model to `checkpoints/osra/main/etth1/best_model.pt`.

### Step 3 — Evaluation

Evaluate the diffusion baseline (no OSRA):

```bash
python evaluate.py --dataset etth1 --mode diff_train
```

Evaluate the OSRA model:

```bash
python evaluate.py --dataset etth1 --mode osra
```

Results are written to `evaluation_results/diff_train/` or `evaluation_results/osra/`.

## Evaluation Metrics

Eight metrics are reported:

| Category | Metrics |
|----------|---------|
| Fidelity & distribution | MSE, MAE, DTW, KLD |
| Semantic alignment | Trend, Volatility, Peak, Avg |