import torch
import numpy as np
import os
import sys
import argparse
import json
import pickle
import random
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from diffusion_core.scheduler import GaussianDiffusion
from diffusion_core.model import TimeSeriesDiffuserV2

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id=0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def generate_sequences(model, diffusion, test_x, test_emb, test_mask, device, batch_size=64):
    model.eval()
    all_generated = []
    dataset = TensorDataset(
        torch.from_numpy(test_x).float(),
        torch.from_numpy(test_emb).float(),
        torch.from_numpy(test_mask).bool()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_x, batch_emb, batch_mask in tqdm(loader, desc='Generating sequences'):
            batch_x = batch_x.to(device)
            batch_emb = batch_emb.to(device)
            batch_mask = batch_mask.to(device)
            B, seq_len = batch_x.shape[0], batch_x.shape[1]
            generated = diffusion.sample(batch_emb, batch_mask, shape=(B, seq_len, 1))
            all_generated.append(generated.cpu().numpy())

    return np.concatenate(all_generated, axis=0)


def calc_dtw_distance(real, generated):
    real_series = real.squeeze(-1)
    gen_series = generated.squeeze(-1)
    distances = []

    for r, g in zip(real_series, gen_series):
        n, m = len(r), len(g)
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        dp[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(r[i - 1] - g[j - 1])
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

        distances.append(dp[n, m] / (n + m))

    return float(np.mean(distances))


def calc_stepwise_marginal_kld(real, generated):
    real_seq = real.squeeze(-1)
    gen_seq = generated.squeeze(-1)

    mu_r = np.mean(real_seq, axis=0)
    var_r = np.var(real_seq, axis=0) + 1e-8
    mu_g = np.mean(gen_seq, axis=0)
    var_g = np.var(gen_seq, axis=0) + 1e-8

    step_kld = 0.5 * (np.log(var_g / var_r) + (var_r + (mu_r - mu_g) ** 2) / var_g - 1.0)
    return float(np.mean(step_kld))


def extract_features(series):
    B, seq_len, _ = series.shape
    s = series.squeeze(-1)
    x = np.arange(seq_len, dtype=np.float32)
    x_mean = x.mean()
    slopes, vols, peaks = [], [], []

    for i in range(B):
        y = s[i]
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum() + 1e-8
        slope = num / den

        val_range = y.max() - y.min() + 1e-8
        norm_slope = slope * seq_len / val_range
        slopes.append(norm_slope)

        trend_line = slope * x + (y_mean - slope * x_mean)
        detrended = y - trend_line
        vol = detrended.std() / (y.std() + 1e-8)
        vols.append(vol)

        peaks.append(y.argmax() / seq_len)

    return np.array(slopes), np.array(vols), np.array(peaks)


def classify_trend(s):
    if s >= 0.15:
        return 'strong_up'
    if s >= 0.05:
        return 'moderate_up'
    if s > -0.05:
        return 'stable'
    if s > -0.15:
        return 'moderate_down'
    return 'strong_down'


def classify_volatility(v):
    if v < 0.5:
        return 'low'
    if v < 0.8:
        return 'medium'
    return 'high'


def classify_peak(p):
    if p < 1 / 3:
        return 'early'
    if p < 2 / 3:
        return 'middle'
    return 'late'


def calc_semantic_accuracy(generated, meta_list):
    slopes, vols, peaks = extract_features(generated)
    B = len(meta_list)
    tc = vc = pc = 0

    for i in range(B):
        m = meta_list[i]
        if classify_trend(slopes[i]) == m.get('trend', ''):
            tc += 1
        if classify_volatility(vols[i]) == m.get('volatility', ''):
            vc += 1
        if classify_peak(peaks[i]) == m.get('peak_location', ''):
            pc += 1

    return tc / B, vc / B, pc / B


def resolve_eval_setup(root, dataset, mode, ckpt_dir, experiment_name, tag, ckpt_name, result_root):
    result_base = Path(result_root) if result_root else root / 'evaluation_results'

    if mode == 'diff_train':
        base_dir = Path(ckpt_dir) if ckpt_dir else root / 'checkpoints' / 'diff_train' / dataset
        ckpt_path = base_dir / ckpt_name
        model_label = 'Diffusion (No OSRA)'
        result_tag = 'diff_train'
        result_dir = result_base / 'diff_train'
    elif mode == 'osra':
        base_dir = (
            Path(ckpt_dir)
            if ckpt_dir
            else root / 'checkpoints' / 'osra' / experiment_name / dataset
        )
        ckpt_path = base_dir / ckpt_name
        model_label = 'OSRA'
        result_tag = 'osra'
        result_dir = result_base / 'osra'
    else:
        raise ValueError(f"Unknown mode={mode}. Available: diff_train, osra")

    if tag:
        result_tag = f'{result_tag}_{tag}'

    return ckpt_path, model_label, result_tag, result_dir


def evaluate(args):
    set_seed(args.seed)
    device = get_device(args.gpu)
    root = Path(__file__).resolve().parent
    dataset = args.dataset.lower()
    data_dir = Path(args.data_dir)

    ckpt_path, model_label, result_tag, result_dir = resolve_eval_setup(
        root, dataset, args.mode, args.ckpt_dir, args.experiment_name,
        args.tag, args.ckpt_name, args.result_root
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset}  [{model_label}]  [seed={args.seed}]")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'=' * 60}")

    all_x = np.load(data_dir / dataset / f'{dataset}_X.npy')
    all_emb = np.load(data_dir / dataset / f'{dataset}_emb.npy')
    all_emb_mask = np.load(data_dir / dataset / f'{dataset}_emb_mask.npy')
    with open(data_dir / dataset / f'{dataset}_meta.pkl', 'rb') as f:
        all_meta = pickle.load(f)

    train_indices = np.load(data_dir / dataset / f'{dataset}_train_indices.npy')
    all_indices = np.arange(len(all_x))
    train_set = set(train_indices.tolist())
    test_indices = np.array([i for i in all_indices if i not in train_set])

    test_x = all_x[test_indices]
    test_emb = all_emb[test_indices]
    test_mask = all_emb_mask[test_indices]
    test_meta = [all_meta[i] for i in test_indices]

    print(f"Test set size: {len(test_x)}  shape: {test_x.shape}")

    model = TimeSeriesDiffuserV2(seq_len=96, text_dim=768, model_dim=256, num_layers=4, nhead=8).to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    diffusion = GaussianDiffusion(model, device=device)

    print("\nGenerating sequences...")
    generated = generate_sequences(model, diffusion, test_x, test_emb, test_mask, device)

    mse = float(np.mean((generated - test_x) ** 2))
    mae = float(np.mean(np.abs(generated - test_x)))
    dtw = float(calc_dtw_distance(test_x, generated))
    kld = float(calc_stepwise_marginal_kld(test_x, generated))

    trend, volatility, peak = calc_semantic_accuracy(generated, test_meta)
    avg = (trend + volatility + peak) / 3

    print(f"\n{'=' * 60}")
    print('Fidelity & Distribution Alignment')
    print(f"{'=' * 60}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"DTW: {dtw:.6f}")
    print(f"KLD: {kld:.6f}")

    print(f"\n{'=' * 60}")
    print('Semantic Alignment')
    print(f"{'=' * 60}")
    print(f"Trend:      {trend:.4f} ({trend * 100:.2f}%)")
    print(f"Volatility: {volatility:.4f} ({volatility * 100:.2f}%)")
    print(f"Peak:       {peak:.4f} ({peak * 100:.2f}%)")
    print(f"Avg:        {avg:.4f} ({avg * 100:.2f}%)")

    result_path = result_dir / f'{dataset}_{result_tag}.json'
    results = {
        'dataset': dataset,
        'model': model_label,
        'seed': args.seed,
        'checkpoint': str(ckpt_path),
        'checkpoint_name': args.ckpt_name,
        'MSE': mse,
        'MAE': mae,
        'DTW': dtw,
        'KLD': kld,
        'Trend': float(trend),
        'Volatility': float(volatility),
        'Peak': float(peak),
        'Avg': float(avg),
    }
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved: {result_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='osra', choices=['diff_train', 'osra'])
    parser.add_argument('--experiment_name', type=str, default='main')
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--ckpt_name', type=str, default='best_model.pt')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--result_root', type=str, default='')
    args = parser.parse_args()
    evaluate(args)
