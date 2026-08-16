import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_core.scheduler import GaussianDiffusion
from diffusion_core.model import TimeSeriesDiffuser

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


# =============================================================================
# Basic utilities
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def path_from_root(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root / p


def safe_tag(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    return (
        s.replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(".", "")
    )


# =============================================================================
# Protocol validation
# =============================================================================

def validate_stats(
    data_root: Path,
    dataset: str,
    protocol: str = "legacy_random",
    allow_nonzero_fallback: bool = False,
) -> Dict:
    """
    protocol = legacy_random:
        Original SemAlign-TS data protocol:
        - random window split
        - fixed volatility thresholds 0.5 / 0.8
        - stats.json may not contain text_generation/raw_overlap/volatility_thresholds

    protocol = chrono:
        New chronological blocked protocol:
        - raw_overlap must be zero
        - volatility_label_mode must be train_quantile
        - q_low/q_high must be saved in stats.json
    """
    stats_path = data_root / dataset / f"{dataset}_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")

    stats = load_json(stats_path)

    if protocol == "legacy_random":
        return stats

    if protocol == "raw_disjoint":
        split = stats.get("split", {})
        if split.get("protocol") != "raw_disjoint":
            raise ValueError(
                f"{dataset}: stats do not declare split.protocol=raw_disjoint"
            )
        raw_overlap = split.get("raw_overlap", {})
        required_pairs = ("train_val", "train_test", "val_test")
        bad_overlap = {
            pair: raw_overlap.get(pair, {}).get("overlap_count", None)
            for pair in required_pairs
            if int(raw_overlap.get(pair, {}).get("overlap_count", -1)) != 0
        }
        if bad_overlap:
            raise ValueError(f"{dataset}: raw overlap is not zero: {bad_overlap}")
        return stats

    if protocol != "chrono":
        raise ValueError(f"Unknown protocol: {protocol}")

    fallback_count = int(stats.get("text_generation", {}).get("fallback_count", -1))
    if fallback_count != 0 and not allow_nonzero_fallback:
        raise ValueError(
            f"{dataset}: fallback_count={fallback_count}. "
            "Final chrono evaluation should use zero-fallback processed data."
        )

    raw_overlap = stats.get("split", {}).get("raw_overlap", {})
    bad_overlap = {
        k: v.get("overlap_count")
        for k, v in raw_overlap.items()
        if int(v.get("overlap_count", 1)) != 0
    }
    if bad_overlap:
        raise ValueError(f"{dataset}: raw overlap is not zero: {bad_overlap}")

    vol_mode = stats.get("volatility_label_mode", None)
    if vol_mode != "train_quantile":
        raise ValueError(
            f"{dataset}: volatility_label_mode={vol_mode}, expected train_quantile"
        )

    vt = stats.get("volatility_thresholds", {})
    if "q_low" not in vt or "q_high" not in vt:
        raise ValueError(
            f"{dataset}: missing volatility_thresholds.q_low/q_high in stats.json"
        )

    return stats


def parse_volatility_thresholds(stats: Dict, protocol: str = "legacy_random") -> Tuple[float, float]:
    if protocol in {"legacy_random", "raw_disjoint"}:
        return 0.5, 0.8

    vt = stats["volatility_thresholds"]
    q_low = float(vt["q_low"])
    q_high = float(vt["q_high"])
    if not q_low < q_high:
        raise ValueError(f"Invalid volatility thresholds: q_low={q_low}, q_high={q_high}")
    return q_low, q_high


# =============================================================================
# Dataset
# =============================================================================

class EvalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        dataset: str,
        split: str = "test",
        max_samples: int = 0,
        condition_mode: str = "correct",
        condition_seed: int = 123,
        emb_suffix: str = "",
    ):
        self.data_root = data_root
        self.dataset = dataset
        self.split = split
        self.condition_mode = condition_mode
        self.condition_seed = int(condition_seed)
        self.emb_suffix = emb_suffix

        root = data_root / dataset

        x_path = root / f"{dataset}_X.npy"
        emb_path = root / f"{dataset}_emb{emb_suffix}.npy"
        mask_path = root / f"{dataset}_emb_mask{emb_suffix}.npy"
        idx_path = root / f"{dataset}_{split}_indices.npy"
        meta_path = root / f"{dataset}_meta.pkl"

        for p in [x_path, emb_path, mask_path, idx_path, meta_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        self.x = np.load(x_path, mmap_mode="r")
        self.emb = np.load(emb_path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        self.indices = np.load(idx_path)

        if max_samples and max_samples > 0:
            self.indices = self.indices[: int(max_samples)]

        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

        if len(self.indices) == 0:
            raise ValueError(f"{dataset} {split} split is empty")

        self.condition_indices = self.indices.copy()

        if condition_mode == "correct":
            pass

        elif condition_mode == "shuffle":
            rng = np.random.RandomState(self.condition_seed)
            self.condition_indices = self.condition_indices.copy()
            rng.shuffle(self.condition_indices)

        elif condition_mode == "zero":
            # Zero text embedding, but keep the original attention mask.
            # This avoids all-masked cross-attention in transformer implementations.
            self.condition_indices = self.indices.copy()

        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, item):
        raw_idx = int(self.indices[item])
        cond_idx = int(self.condition_indices[item])

        x = torch.from_numpy(np.array(self.x[raw_idx], dtype=np.float32, copy=True))

        if self.condition_mode == "zero":
            emb_arr = np.zeros_like(np.array(self.emb[raw_idx], dtype=np.float32, copy=True))
            mask_arr = np.array(self.mask[raw_idx], dtype=np.bool_, copy=True)
        else:
            emb_arr = np.array(self.emb[cond_idx], dtype=np.float32, copy=True)
            mask_arr = np.array(self.mask[cond_idx], dtype=np.bool_, copy=True)

        emb = torch.from_numpy(emb_arr)
        mask = torch.from_numpy(mask_arr)

        return x, emb, mask, raw_idx, cond_idx

    def get_meta_batch(self, raw_indices):
        return [self.meta[int(i)] for i in raw_indices]


# =============================================================================
# Model / checkpoint
# =============================================================================

def load_model_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)


def resolve_eval_setup(
    root: Path,
    dataset: str,
    mode: str,
    ckpt_dir: str,
    tag: str,
    ckpt_name: str,
    result_root: str,
    experiment_name: str,
    seed: int,
    seed_folder: str,
    grpo_seed_folder: str,
    pretrain_root: str,
    grpo_root: str,
    dpo_root: str,
    condition_mode: str,
    emb_suffix: str,
):
    if not seed_folder:
        seed_folder = f"seed{seed}"
    if not grpo_seed_folder:
        grpo_seed_folder = f"{seed_folder}_static"

    result_base = path_from_root(root, result_root) if result_root else root / "evaluation_results_seeded"

    if mode == "pretrain":
        if ckpt_dir:
            base_dir = path_from_root(root, ckpt_dir)
        else:
            base_dir = path_from_root(root, pretrain_root) / seed_folder / dataset

        model_label = "pretrain_no_rl"
        result_dir = result_base / seed_folder / "pretrain"

    elif mode in {"grpo", "osra"}:
        if ckpt_dir:
            base_dir = path_from_root(root, ckpt_dir)
        else:
            base_dir = path_from_root(root, grpo_root) / grpo_seed_folder / experiment_name / dataset

        model_label = f"semaalign_ts_{experiment_name}"
        result_dir = result_base / seed_folder / "osra" / experiment_name

    elif mode == "dpo":
        if ckpt_dir:
            base_dir = path_from_root(root, ckpt_dir)
        else:
            base_dir = path_from_root(root, dpo_root) / seed_folder / experiment_name / dataset

        model_label = f"dpo_{experiment_name}"
        result_dir = result_base / seed_folder / "dpo" / experiment_name

    else:
        raise ValueError(f"Unknown mode: {mode}")

    ckpt_path = base_dir / ckpt_name

    # DPO fallback: allow dpo_checkpoints_seeded/seedXXXX/dataset/best_model.pt
    if mode == "dpo" and not ckpt_path.exists() and not ckpt_dir:
        fallback_dir = path_from_root(root, dpo_root) / seed_folder / dataset
        fallback_path = fallback_dir / ckpt_name
        if fallback_path.exists():
            ckpt_path = fallback_path

    result_parts = [model_label]
    if tag:
        result_parts.append(safe_tag(tag))
    if condition_mode != "correct":
        result_parts.append(f"cond-{safe_tag(condition_mode)}")
    if emb_suffix:
        result_parts.append(f"emb-{safe_tag(emb_suffix)}")

    result_tag = "_".join([p for p in result_parts if p])
    return ckpt_path, model_label, result_tag, result_dir, seed_folder, grpo_seed_folder


# =============================================================================
# Generation
# =============================================================================

def generate_sequences(
    model,
    diffusion,
    eval_set: EvalDataset,
    device: torch.device,
    batch_size: int = 64,
    sampler: str = "ddim",
    ddim_steps: int = 50,
    eta: float = 0.0,
):
    model.eval()

    loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_real = []
    all_generated = []
    all_raw_indices = []
    all_cond_indices = []

    with torch.no_grad():
        for batch_x, batch_emb, batch_mask, batch_raw_idx, batch_cond_idx in tqdm(loader, desc="Generating"):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            bsz, seq_len = batch_x.shape[0], batch_x.shape[1]

            if sampler == "ddpm":
                generated = diffusion.sample(
                    batch_emb,
                    batch_mask,
                    shape=(bsz, seq_len, 1),
                )
            elif sampler == "ddim":
                generated = diffusion.ddim_sample(
                    batch_emb,
                    batch_mask,
                    shape=(bsz, seq_len, 1),
                    ddim_steps=ddim_steps,
                    eta=eta,
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            all_real.append(batch_x.cpu().numpy())
            all_generated.append(generated.cpu().numpy())
            all_raw_indices.extend([int(x) for x in batch_raw_idx.cpu().tolist()])
            all_cond_indices.extend([int(x) for x in batch_cond_idx.cpu().tolist()])

    real = np.concatenate(all_real, axis=0)
    generated = np.concatenate(all_generated, axis=0)

    raw_indices = np.array(all_raw_indices, dtype=np.int64)
    cond_indices = np.array(all_cond_indices, dtype=np.int64)

    target_meta = eval_set.get_meta_batch(raw_indices)
    condition_meta = eval_set.get_meta_batch(cond_indices)

    return real, generated, target_meta, condition_meta, raw_indices, cond_indices


# =============================================================================
# Metrics
# =============================================================================

def denormalize_batch(normalized_series: torch.Tensor, meta_list, device: torch.device) -> torch.Tensor:
    min_vals = torch.tensor(
        [m["min_value"] for m in meta_list],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)

    max_vals = torch.tensor(
        [m["max_value"] for m in meta_list],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)

    return (normalized_series + 1.0) / 2.0 * (max_vals - min_vals) + min_vals


def calc_dtw_distance(real: np.ndarray, generated: np.ndarray) -> float:
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


def calc_gaussian_kld(real: np.ndarray, generated: np.ndarray) -> float:
    real_seq = real.squeeze(-1)
    gen_seq = generated.squeeze(-1)

    mu_r = np.mean(real_seq, axis=0)
    var_r = np.var(real_seq, axis=0) + 1e-8

    mu_g = np.mean(gen_seq, axis=0)
    var_g = np.var(gen_seq, axis=0) + 1e-8

    step_kld = 0.5 * (
        np.log(var_g / var_r)
        + (var_r + (mu_r - mu_g) ** 2) / var_g
        - 1.0
    )

    return float(np.mean(step_kld))



def marginal_distribution_difference(
    real: np.ndarray,
    generated: np.ndarray,
    n_bins: int = 50,
) -> float:
    """Marginal Distribution Difference (MDD), lower is better.

    This is intentionally identical to the evaluator used by the external
    SemAlign baseline adapters (TSGBench-compatible NumPy implementation):
      - exclude t=0;
      - build real-data histograms independently for each time step/dimension;
      - evaluate generated density on exactly the same real-data bins;
      - average the absolute density difference across bins/time/dimensions.

    MDD is a population-level metric, not a paired-sample metric.
    """
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)

    if real.ndim == 2:
        real = real[..., None]
    if generated.ndim == 2:
        generated = generated[..., None]
    if real.ndim != 3 or generated.ndim != 3:
        raise ValueError(
            f"MDD expects [N,L,D], got real={real.shape}, "
            f"generated={generated.shape}"
        )
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError(
            f"MDD shape mismatch: real={real.shape}, generated={generated.shape}"
        )

    # Match the shared baseline evaluator / TSGBench convention.
    real = real[:, 1:, :]
    generated = generated[:, 1:, :]

    losses = []
    for dim in range(real.shape[2]):
        for t in range(real.shape[1]):
            r = real[:, t, dim]
            g = generated[:, t, dim]

            a = float(np.min(r))
            b = float(np.max(r))
            if b == a:
                b = a + 1e-5

            edges = np.linspace(a, b, n_bins + 1, dtype=np.float64)
            delta = float(edges[1] - edges[0])

            real_count, _ = np.histogram(r, bins=edges)
            gen_count, _ = np.histogram(g, bins=edges)

            real_density = real_count.astype(np.float64) / delta / float(len(r))
            gen_density = gen_count.astype(np.float64) / delta / float(len(g))
            losses.append(float(np.mean(np.abs(gen_density - real_density))))

    return float(np.mean(losses))


def _tsgbench_acf(
    x: np.ndarray,
    max_lag: int = 64,
    eps: float = 1e-12,
) -> np.ndarray:
    """Stationary ACF definition used by the shared baseline evaluator."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f"ACF expects [N,L,D], got {x.shape}")

    centered = x - np.mean(x, axis=(0, 1), keepdims=True)
    var = np.mean(centered ** 2, axis=(0, 1))
    var = np.maximum(var, eps)

    max_lag = min(int(max_lag), x.shape[1])
    values = []
    for lag in range(max_lag):
        if lag == 0:
            products = centered ** 2
        else:
            products = centered[:, lag:, :] * centered[:, :-lag, :]
        cov = np.mean(products, axis=(0, 1))
        values.append(cov / var)

    return np.stack(values, axis=0)


def autocorrelation_difference(
    real: np.ndarray,
    generated: np.ndarray,
    max_lag: int = 64,
) -> float:
    """AutoCorrelation Difference (ACD), lower is better.

    Matches the shared baseline evaluator: L2 difference over lag for each
    dimension, then mean over dimensions.
    """
    real_acf = _tsgbench_acf(real, max_lag=max_lag)
    generated_acf = _tsgbench_acf(generated, max_lag=max_lag)
    per_dim = np.sqrt(np.sum((generated_acf - real_acf) ** 2, axis=0))
    return float(np.mean(per_dim))


def extract_features(series: np.ndarray):
    batch_size, seq_len, _ = series.shape
    s = series.squeeze(-1)

    x = np.arange(seq_len, dtype=np.float32)
    x_mean = x.mean()

    slopes = []
    vols = []
    peaks = []

    for i in range(batch_size):
        y = s[i]
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum() + 1e-8

        slope = numerator / denominator
        value_range = y.max() - y.min() + 1e-8
        normalized_slope = slope * seq_len / value_range
        slopes.append(normalized_slope)

        trend_line = slope * x + (y_mean - slope * x_mean)
        detrended = y - trend_line
        normalized_volatility = detrended.std() / (y.std() + 1e-8)
        vols.append(normalized_volatility)

        peaks.append(y.argmax() / seq_len)

    return np.array(slopes), np.array(vols), np.array(peaks)


def classify_trend(slope: float) -> str:
    if slope >= 0.15:
        return "strong_up"
    if slope >= 0.05:
        return "moderate_up"
    if slope > -0.05:
        return "stable"
    if slope > -0.15:
        return "moderate_down"
    return "strong_down"


def classify_volatility(volatility: float, q_low: float, q_high: float) -> str:
    if volatility < q_low:
        return "low"
    if volatility < q_high:
        return "medium"
    return "high"


def classify_peak(peak: float) -> str:
    if peak < 1.0 / 3.0:
        return "early"
    if peak < 2.0 / 3.0:
        return "middle"
    return "late"


def collapse_trend_3(label: str) -> str:
    if label in ["strong_up", "moderate_up"]:
        return "Up"
    if label in ["strong_down", "moderate_down"]:
        return "Down"
    return "Stable"


def predict_semantic_labels(generated: np.ndarray, q_low: float, q_high: float) -> List[Dict]:
    slopes, vols, peaks = extract_features(generated)

    preds = []
    for i in range(generated.shape[0]):
        trend_pred = classify_trend(float(slopes[i]))
        vol_pred = classify_volatility(float(vols[i]), q_low=q_low, q_high=q_high)
        peak_pred = classify_peak(float(peaks[i]))

        preds.append({
            "trend_pred": trend_pred,
            "vol_pred": vol_pred,
            "peak_pred": peak_pred,
            "trend_pred_3": collapse_trend_3(trend_pred),
            "slope_value": float(slopes[i]),
            "volatility_value": float(vols[i]),
            "peak_ratio": float(peaks[i]),
        })

    return preds


def calc_semantic_accuracy_from_predictions(preds: List[Dict], meta_list) -> Dict:
    trend_hits = []
    vol_hits = []
    peak_hits = []
    joint_hits = []

    for pred, meta in zip(preds, meta_list):
        trend_true = meta.get("trend", "")
        vol_true = meta.get("volatility", "")
        peak_true = meta.get("peak_location", "")

        trend_ok = pred["trend_pred"] == trend_true
        vol_ok = pred["vol_pred"] == vol_true
        peak_ok = pred["peak_pred"] == peak_true

        trend_hits.append(float(trend_ok))
        vol_hits.append(float(vol_ok))
        peak_hits.append(float(peak_ok))
        joint_hits.append(float(trend_ok and vol_ok and peak_ok))

    trend_acc = float(np.mean(trend_hits))
    vol_acc = float(np.mean(vol_hits))
    peak_acc = float(np.mean(peak_hits))
    joint_acc_3 = float(np.mean(joint_hits))

    return {
        "trend_acc": trend_acc,
        "vol_acc": vol_acc,
        "peak_acc": peak_acc,
        "avg_semantic": float((trend_acc + vol_acc + peak_acc) / 3.0),
        "joint_acc_3": joint_acc_3,
    }


def build_detail_rows(
    preds: List[Dict],
    target_meta,
    condition_meta,
    raw_indices: np.ndarray,
    cond_indices: np.ndarray,
) -> List[Dict]:
    rows = []

    for i, pred in enumerate(preds):
        t = target_meta[i]
        c = condition_meta[i]

        target_trend = t.get("trend", "")
        target_vol = t.get("volatility", "")
        target_peak = t.get("peak_location", "")

        cond_trend = c.get("trend", "")
        cond_vol = c.get("volatility", "")
        cond_peak = c.get("peak_location", "")

        target_trend_ok = pred["trend_pred"] == target_trend
        target_vol_ok = pred["vol_pred"] == target_vol
        target_peak_ok = pred["peak_pred"] == target_peak

        prompt_trend_ok = pred["trend_pred"] == cond_trend
        prompt_vol_ok = pred["vol_pred"] == cond_vol
        prompt_peak_ok = pred["peak_pred"] == cond_peak

        row = {
            "i": int(i),
            "raw_idx": int(raw_indices[i]),
            "cond_idx": int(cond_indices[i]),

            "target_trend": target_trend,
            "target_trend_3": collapse_trend_3(target_trend),
            "target_volatility": target_vol,
            "target_peak": target_peak,

            "prompt_trend": cond_trend,
            "prompt_trend_3": collapse_trend_3(cond_trend),
            "prompt_volatility": cond_vol,
            "prompt_peak": cond_peak,

            "pred_trend": pred["trend_pred"],
            "pred_trend_3": pred["trend_pred_3"],
            "pred_volatility": pred["vol_pred"],
            "pred_peak": pred["peak_pred"],

            "target_trend_ok": bool(target_trend_ok),
            "target_vol_ok": bool(target_vol_ok),
            "target_peak_ok": bool(target_peak_ok),
            "target_joint3_ok": bool(target_trend_ok and target_vol_ok and target_peak_ok),

            "prompt_trend_ok": bool(prompt_trend_ok),
            "prompt_vol_ok": bool(prompt_vol_ok),
            "prompt_peak_ok": bool(prompt_peak_ok),
            "prompt_joint3_ok": bool(prompt_trend_ok and prompt_vol_ok and prompt_peak_ok),

            "slope_value": pred["slope_value"],
            "volatility_value": pred["volatility_value"],
            "peak_ratio": pred["peak_ratio"],
        }

        rows.append(row)

    return rows


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(args):
    set_seed(args.seed)

    device = get_device(args.gpu)
    root = Path(__file__).resolve().parent

    dataset = args.dataset.lower()
    data_root = path_from_root(root, args.data_dir) if args.data_dir else root / "processed_data"

    stats = validate_stats(
        data_root=data_root,
        dataset=dataset,
        protocol=args.protocol,
        allow_nonzero_fallback=args.allow_nonzero_fallback,
    )
    q_low, q_high = parse_volatility_thresholds(stats, protocol=args.protocol)

    ckpt_path, model_label, result_tag, result_dir, seed_folder, grpo_seed_folder = resolve_eval_setup(
        root=root,
        dataset=dataset,
        mode=args.mode,
        ckpt_dir=args.ckpt_dir,
        tag=args.tag,
        ckpt_name=args.ckpt_name,
        result_root=args.result_root,
        experiment_name=args.experiment_name,
        seed=args.seed,
        seed_folder=args.seed_folder,
        grpo_seed_folder=args.grpo_seed_folder,
        pretrain_root=args.pretrain_root,
        grpo_root=args.grpo_root,
        dpo_root=args.dpo_root,
        condition_mode=args.condition_mode,
        emb_suffix=args.emb_suffix,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    if args.split != "test":
        result_tag = f"{result_tag}_split-{args.split}"

    print("\n" + "=" * 80)
    print(f"Dataset: {dataset} | model={model_label} | seed={args.seed}")
    print(f"protocol: {args.protocol}")
    print(f"seed_folder: {seed_folder} | grpo_seed_folder: {grpo_seed_folder}")
    print(f"checkpoint: {ckpt_path}")
    print(f"data_root: {data_root}")
    print(f"split: {args.split}_indices")
    print(f"condition_mode: {args.condition_mode} | condition_seed={args.condition_seed}")
    print(f"emb_suffix: '{args.emb_suffix}'")
    print(f"sampler: {args.sampler} | ddim_steps={args.ddim_steps} | eta={args.eta}")
    print(f"batch_size={args.batch_size} | max_eval_samples={args.max_eval_samples}")
    print(f"volatility thresholds: q_low={q_low:.6f}, q_high={q_high:.6f}")
    print("=" * 80)

    eval_set = EvalDataset(
        data_root=data_root,
        dataset=dataset,
        split=args.split,
        max_samples=args.max_eval_samples,
        condition_mode=args.condition_mode,
        condition_seed=args.condition_seed,
        emb_suffix=args.emb_suffix,
    )

    sample_x, sample_emb, _, _, _ = eval_set[0]
    seq_len = int(sample_x.shape[0])
    text_dim = int(sample_emb.shape[-1])

    print(f"{args.split.capitalize()} samples: {len(eval_set)} | x_shape={tuple(sample_x.shape)} | text_dim={text_dim}")

    model = TimeSeriesDiffuser(
        seq_len=seq_len,
        input_dim=1,
        text_dim=text_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_model_state(model, ckpt_path, device)
    print(f"Loaded: {ckpt_path}")

    diffusion = GaussianDiffusion(
        model,
        device=str(device),
        timesteps=args.timesteps,
        schedule=args.beta_schedule,
    )

    print("\nGenerating sequences...")
    test_x, generated, target_meta, condition_meta, raw_indices, cond_indices = generate_sequences(
        model=model,
        diffusion=diffusion,
        eval_set=eval_set,
        device=device,
        batch_size=args.batch_size,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
    )

    gen_tensor = torch.from_numpy(generated).float().to(device)
    real_tensor = torch.from_numpy(test_x).float().to(device)

    gen_real = denormalize_batch(gen_tensor, target_meta, device).cpu().numpy()
    real_real = denormalize_batch(real_tensor, target_meta, device).cpu().numpy()

    print("\n" + "=" * 80)
    print("Fidelity and distribution metrics against paired target")
    print("=" * 80)

    mse_norm = float(np.mean((generated - test_x) ** 2))
    mae_norm = float(np.mean(np.abs(generated - test_x)))

    mse_real = float(np.mean((gen_real - real_real) ** 2))
    mae_real = float(np.mean(np.abs(gen_real - real_real)))

    dtw_norm = None if args.skip_dtw else float(calc_dtw_distance(test_x, generated))
    dtw_real = None if args.skip_dtw else float(calc_dtw_distance(real_real, gen_real))

    # Paper-facing population metrics are computed in the same normalized
    # space and with the same implementation as every external baseline.
    mdd = float(marginal_distribution_difference(test_x, generated, n_bins=50))
    acd = float(autocorrelation_difference(test_x, generated, max_lag=64))

    # Historical Gaussian KLD is retained only as a diagnostic/backward-
    # compatibility field. It is not one of the nine paper-facing metrics.
    kld_norm = float(calc_gaussian_kld(test_x, generated))
    kld_real = float(calc_gaussian_kld(real_real, gen_real))

    print(
        f"[Normalized paired] MSE: {mse_norm:.6f} | MAE: {mae_norm:.6f} | "
        f"DTW: {dtw_norm if dtw_norm is not None else 'SKIP'}"
    )
    print(f"[Normalized population] MDD: {mdd:.6f} | ACD: {acd:.6f}")
    print(
        f"[Diagnostic only] Gaussian KLD(norm): {kld_norm:.6f} | "
        f"KLD(real): {kld_real:.6f}"
    )

    print("\n" + "=" * 80)
    print("Semantic alignment")
    print("=" * 80)

    preds = predict_semantic_labels(generated, q_low=q_low, q_high=q_high)

    sem_prompt = calc_semantic_accuracy_from_predictions(preds, condition_meta)
    sem_target = calc_semantic_accuracy_from_predictions(preds, target_meta)

    # Backward-compatible top-level semantic metrics:
    # For normal evaluation condition_mode=correct, prompt and target are identical.
    # For QR5 shuffled condition, top-level semantic metrics mean prompt-following.
    sem_main = sem_prompt

    print("[Against input prompt / condition labels]")
    print(f"Trend Acc:      {sem_prompt['trend_acc']:.4f} ({sem_prompt['trend_acc'] * 100:.2f}%)")
    print(f"Volatility Acc: {sem_prompt['vol_acc']:.4f} ({sem_prompt['vol_acc'] * 100:.2f}%)")
    print(f"Peak Acc:       {sem_prompt['peak_acc']:.4f} ({sem_prompt['peak_acc'] * 100:.2f}%)")
    print(f"Avg Semantic:   {sem_prompt['avg_semantic']:.4f} ({sem_prompt['avg_semantic'] * 100:.2f}%)")
    print(f"Joint Acc-3:    {sem_prompt['joint_acc_3']:.4f} ({sem_prompt['joint_acc_3'] * 100:.2f}%)")

    print("\n[Against paired target labels]")
    print(f"Trend Acc:      {sem_target['trend_acc']:.4f} ({sem_target['trend_acc'] * 100:.2f}%)")
    print(f"Volatility Acc: {sem_target['vol_acc']:.4f} ({sem_target['vol_acc'] * 100:.2f}%)")
    print(f"Peak Acc:       {sem_target['peak_acc']:.4f} ({sem_target['peak_acc'] * 100:.2f}%)")
    print(f"Avg Semantic:   {sem_target['avg_semantic']:.4f} ({sem_target['avg_semantic'] * 100:.2f}%)")
    print(f"Joint Acc-3:    {sem_target['joint_acc_3']:.4f} ({sem_target['joint_acc_3'] * 100:.2f}%)")

    fidelity_terms = [
        1.0 / (1.0 + mse_norm),
        1.0 / (1.0 + mae_norm),
        1.0 / (1.0 + kld_norm),
    ]
    if dtw_norm is not None:
        fidelity_terms.append(1.0 / (1.0 + dtw_norm))

    fidelity_score = float(np.mean(fidelity_terms))
    overall = float(fidelity_score * 0.6 + sem_main["avg_semantic"] * 0.4)

    print(f"\nOverall: {overall:.4f}")

    detail_path = None
    if args.save_details:
        detail_rows = build_detail_rows(
            preds=preds,
            target_meta=target_meta,
            condition_meta=condition_meta,
            raw_indices=raw_indices,
            cond_indices=cond_indices,
        )

        detail_path = result_dir / f"{dataset}_{result_tag}_seed{args.seed}_details.jsonl"
        with open(detail_path, "w", encoding="utf-8") as f:
            for row in detail_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Saved details: {detail_path}")

    result_path = result_dir / f"{dataset}_{result_tag}_seed{args.seed}.json"

    generated_cache_path = None
    if args.save_generated:
        generated_cache_path = result_dir / f"{dataset}_{result_tag}_seed{args.seed}_generated.npz"
        np.savez_compressed(
            generated_cache_path,
            generated=np.asarray(generated, dtype=np.float32),
            raw_indices=np.asarray(raw_indices, dtype=np.int64),
            condition_indices=np.asarray(cond_indices, dtype=np.int64),
        )
        print(f"Saved generated cache: {generated_cache_path}")

    results = {
        "dataset": dataset,
        "model": model_label,
        "mode": args.mode,
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "seed_folder": seed_folder,
        "grpo_seed_folder": grpo_seed_folder,

        "checkpoint": str(ckpt_path),
        "checkpoint_name": args.ckpt_name,

        "protocol": args.protocol,
        "data_root": str(data_root),
        "split": args.split,
        "num_eval_samples": len(eval_set),

        "condition_mode": args.condition_mode,
        "condition_seed": args.condition_seed,
        "emb_suffix": args.emb_suffix,

        "sampler": args.sampler,
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
        "beta_schedule": args.beta_schedule,

        "volatility_thresholds": {
            "q_low": q_low,
            "q_high": q_high,
        },

        "mse_norm": mse_norm,
        "mae_norm": mae_norm,
        "mse_real": mse_real,
        "mae_real": mae_real,
        "dtw_norm": dtw_norm,
        "dtw_real": dtw_real,
        # Nine paper-facing core metrics use normalized-space fidelity.
        "mdd": mdd,
        "acd": acd,
        "distribution_metric_primary": "mdd",
        "paper_core_metrics": [
            "trend_acc", "vol_acc", "peak_acc", "joint_acc_3",
            "mse_norm", "mae_norm", "dtw_norm", "mdd", "acd",
        ],

        # Historical diagnostic fields retained for old scripts/results.
        "kld_norm": kld_norm,
        "kld_real": kld_real,

        # Backward-compatible main semantic metrics.
        # Normal main experiments use condition_mode=correct, so these are unchanged.
        "trend_acc": sem_main["trend_acc"],
        "vol_acc": sem_main["vol_acc"],
        "peak_acc": sem_main["peak_acc"],
        "avg_semantic": sem_main["avg_semantic"],
        "joint_acc_3": sem_main["joint_acc_3"],

        # Explicit prompt-following metrics.
        "prompt_trend_acc": sem_prompt["trend_acc"],
        "prompt_vol_acc": sem_prompt["vol_acc"],
        "prompt_peak_acc": sem_prompt["peak_acc"],
        "prompt_avg_semantic": sem_prompt["avg_semantic"],
        "prompt_joint_acc_3": sem_prompt["joint_acc_3"],

        # Explicit paired-target metrics.
        "target_trend_acc": sem_target["trend_acc"],
        "target_vol_acc": sem_target["vol_acc"],
        "target_peak_acc": sem_target["peak_acc"],
        "target_avg_semantic": sem_target["avg_semantic"],
        "target_joint_acc_3": sem_target["joint_acc_3"],

        "fidelity_score": fidelity_score,
        "overall_score": overall,

        "details_path": str(detail_path) if detail_path is not None else None,
        "generated_cache_path": (
            str(generated_cache_path) if generated_cache_path is not None else None
        ),
        "evaluator_version": "semalign_9core_mdd_acd_v1",
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {result_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--mode", type=str, default="grpo", choices=["pretrain", "osra", "dpo", "grpo"])
    parser.add_argument("--experiment_name", type=str, default="timegrpo_full_g4")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_folder", type=str, default="")
    parser.add_argument("--grpo_seed_folder", type=str, default="")

    parser.add_argument("--protocol", type=str, default="raw_disjoint", choices=["raw_disjoint", "legacy_random", "chrono"])
    parser.add_argument("--data_dir", type=str, default="data/processed")

    parser.add_argument("--pretrain_root", type=str, default="outputs/checkpoints")
    parser.add_argument("--grpo_root", type=str, default="outputs/checkpoints")
    parser.add_argument("--dpo_root", type=str, default="outputs/checkpoints")

    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt")

    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--result_root", type=str, default="outputs/evaluation")

    parser.add_argument(
        "--condition_mode",
        type=str,
        default="correct",
        choices=["correct", "shuffle", "zero"],
        help="correct: matched text; shuffle: mismatched text; zero: zero text embedding",
    )
    parser.add_argument("--condition_seed", type=int, default=123)

    parser.add_argument(
        "--emb_suffix",
        type=str,
        default="",
        help="Use alternative embedding files, e.g. _para1 for dataset_emb_para1.npy",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])

    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--skip_dtw", action="store_true")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument(
        "--save_generated",
        action="store_true",
        help=(
            "Save generated trajectories plus raw/condition indices as a compressed NPZ. "
            "Recommended for the final rerun so future metrics can be added without resampling."
        ),
    )
    parser.add_argument("--allow_nonzero_fallback", action="store_true")

    args = parser.parse_args()
    evaluate(args)