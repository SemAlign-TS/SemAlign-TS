#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_core.model import TimeSeriesDiffuserV2
from diffusion_core.scheduler import GaussianDiffusion
from reward_core.reward import calc_trend_soft_score, classify_trend


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_state(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def lr_factor(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def classify_volatility(value: float) -> str:
    if value < 0.5:
        return "low"
    if value < 0.8:
        return "medium"
    return "high"


def classify_peak(value: float) -> str:
    if value < 1 / 3:
        return "early"
    if value < 2 / 3:
        return "middle"
    return "late"


def extract_features(series):
    batch, length, _ = series.shape
    y = series.squeeze(-1)
    x = torch.arange(length, device=series.device).float()[None, :].expand(batch, -1)
    x_mean = x.mean(1, keepdim=True)
    y_mean = y.mean(1, keepdim=True)
    slope = ((x - x_mean) * (y - y_mean)).sum(1) / (
        ((x - x_mean) ** 2).sum(1) + 1e-8
    )
    value_range = y.max(1).values - y.min(1).values
    normalized_slope = slope * length / (value_range + 1e-8)
    trend_line = slope[:, None] * x + (y_mean - slope[:, None] * x_mean)
    detrended = y - trend_line
    volatility = detrended.std(1) / (y.std(1) + 1e-8)
    peak = y.argmax(1).float() / length
    return normalized_slope, volatility, peak


def alignment_metrics(series, target_meta, target, mse_temperature):
    slope, volatility, peak = extract_features(series)
    trend_hits, vol_hits, peak_hits, trend_soft = [], [], [], []
    for i, meta in enumerate(target_meta):
        sv = float(slope[i].item())
        trend_hits.append(float(classify_trend(sv) == meta.get("trend", "")))
        trend_soft.append(float(calc_trend_soft_score(sv, meta.get("trend", ""))))
        vol_hits.append(
            float(classify_volatility(float(volatility[i].item()))
                  == meta.get("volatility", ""))
        )
        peak_hits.append(
            float(classify_peak(float(peak[i].item()))
                  == meta.get("peak_location", ""))
        )
    mse = F.mse_loss(series, target, reduction="none").mean((1, 2))
    mse_score = torch.exp(-mse / max(mse_temperature, 1e-6))
    return {
        "trend_acc": float(np.mean(trend_hits)),
        "trend_soft_score": float(np.mean(trend_soft)),
        "volatility_acc": float(np.mean(vol_hits)),
        "peak_acc": float(np.mean(peak_hits)),
        "mse_score": float(mse_score.mean().item()),
    }


class PairDataset(Dataset):
    def __init__(self, winner, loser, emb, mask, train_indices, valid_positions):
        self.winner = winner
        self.loser = loser
        self.emb = emb
        self.mask = mask
        self.train_indices = train_indices
        self.valid_positions = valid_positions

    def __len__(self):
        return len(self.valid_positions)

    def __getitem__(self, item):
        pos = int(self.valid_positions[item])
        raw_idx = int(self.train_indices[pos])
        return (
            torch.from_numpy(np.asarray(self.winner[pos])).float(),
            torch.from_numpy(np.asarray(self.loser[pos])).float(),
            torch.from_numpy(np.asarray(self.emb[raw_idx])).float(),
            torch.from_numpy(np.asarray(self.mask[raw_idx])).bool(),
        )


class ValidationDataset(Dataset):
    def __init__(self, x, emb, mask, indices):
        self.x = x
        self.emb = emb
        self.mask = mask
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        raw_idx = int(self.indices[item])
        return (
            torch.from_numpy(np.asarray(self.x[raw_idx])).float(),
            torch.from_numpy(np.asarray(self.emb[raw_idx])).float(),
            torch.from_numpy(np.asarray(self.mask[raw_idx])).bool(),
            item,
        )


@torch.no_grad()
def evaluate_validation(
    diffusion, loader, val_meta, seed, ddim_steps,
    mse_temperature, selection_mse_weight, device,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    totals = {
        "trend_acc": 0.0,
        "trend_soft_score": 0.0,
        "volatility_acc": 0.0,
        "peak_acc": 0.0,
        "mse_score": 0.0,
    }
    count = 0
    for x, emb, mask, positions in loader:
        x = x.to(device)
        emb = emb.to(device)
        mask = mask.to(device)
        batch_meta = [val_meta[int(i)] for i in positions.tolist()]
        noise = torch.randn((len(x), 96, 1), generator=generator, device=device)
        generated = diffusion.ddim_sample(
            emb, mask, shape=(len(x), 96, 1),
            ddim_steps=ddim_steps, eta=0.0, init_noise=noise,
        )
        metrics = alignment_metrics(generated, batch_meta, x, mse_temperature)
        for key in totals:
            totals[key] += metrics[key] * len(x)
        count += len(x)
    result = {key: value / count for key, value in totals.items()}
    result["semantic_score"] = (
        result["trend_acc"] + result["volatility_acc"] + result["peak_acc"]
    ) / 3.0
    result["select_score"] = (
        result["semantic_score"] + selection_mse_weight * result["mse_score"]
    )
    result["num_samples"] = count
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Cached reward-ranked Diffusion-DPO.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--pretrain_dir", required=True)
    p.add_argument("--pair_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--dpo_t_samples", type=int, default=2)
    p.add_argument("--t_low_ratio", type=float, default=0.1)
    p.add_argument("--t_high_ratio", type=float, default=0.9)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mse_temperature", type=float, default=0.5)
    p.add_argument("--selection_mse_weight", type=float, default=0.2)
    p.add_argument("--val_interval", type=int, default=2)
    p.add_argument("--val_batch_size", type=int, default=128)
    p.add_argument("--val_ddim_steps", type=int, default=20)
    p.add_argument("--val_max_samples", type=int, default=0)
    p.add_argument("--validation_seed", type=int, default=91021)
    p.add_argument("--min_epochs", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=4)
    p.add_argument("--early_stop_min_delta", type=float, default=0.001)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--log_interval_batches", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = args.dataset.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    now = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_root = Path(args.data_dir) / dataset
    pretrain_ckpt = Path(args.pretrain_dir) / dataset / "best_model.pt"
    pair_root = Path(args.pair_dir)
    save_root = Path(args.save_dir) / dataset
    save_root.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    metric_log = Path(args.log_dir) / f"{dataset}.jsonl"

    manifest = json.loads((pair_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete pair cache: {pair_root}")
    if int(manifest["seed"]) != args.seed or manifest["dataset"] != dataset:
        raise RuntimeError("Pair cache seed/dataset mismatch.")

    x = np.load(data_root / f"{dataset}_X.npy", mmap_mode="r")
    emb = np.load(data_root / f"{dataset}_emb.npy", mmap_mode="r")
    mask = np.load(data_root / f"{dataset}_emb_mask.npy", mmap_mode="r")
    with (data_root / f"{dataset}_meta.pkl").open("rb") as handle:
        all_meta = pickle.load(handle)
    val_indices = np.load(data_root / f"{dataset}_val_indices.npy")
    if args.val_max_samples > 0 and len(val_indices) > args.val_max_samples:
        rng = np.random.RandomState(args.validation_seed)
        val_indices = np.sort(rng.choice(
            val_indices, args.val_max_samples, replace=False
        ))

    winner = np.load(pair_root / "winner.npy", mmap_mode="r")
    loser = np.load(pair_root / "loser.npy", mmap_mode="r")
    valid_mask = np.load(pair_root / "valid_mask.npy", mmap_mode="r")
    train_indices = np.load(pair_root / "train_indices.npy")
    valid_positions = np.flatnonzero(valid_mask)
    if len(valid_positions) == 0:
        raise RuntimeError("Pair cache has no valid preference pairs.")

    pair_loader = DataLoader(
        PairDataset(winner, loser, emb, mask, train_indices, valid_positions),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        ValidationDataset(x, emb, mask, val_indices),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_meta = [all_meta[int(idx)] for idx in val_indices]

    state = load_state(pretrain_ckpt, device)
    model = TimeSeriesDiffuserV2(
        seq_len=96, text_dim=768, model_dim=256, num_layers=4, nhead=8
    ).to(device)
    model.load_state_dict(state)
    reference = TimeSeriesDiffuserV2(
        seq_len=96, text_dim=768, model_dim=256, num_layers=4, nhead=8
    ).to(device)
    reference.load_state_dict(state)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    diffusion = GaussianDiffusion(model, device=device)
    ref_diffusion = GaussianDiffusion(reference, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    config = vars(args) | {
        "pair_source": "fixed_frozen_reference_cache",
        "same_timestep_and_noise_within_pair": True,
        "anchor_weight": 0.0,
        "num_valid_pairs": len(valid_positions),
        "pair_manifest": manifest,
    }
    (save_root / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    best_score = -float("inf")
    best_epoch = -1
    checks_without_improvement = 0
    t_low = int(args.t_low_ratio * diffusion.timesteps)
    t_high = min(
        diffusion.timesteps,
        max(t_low + 1, int(args.t_high_ratio * diffusion.timesteps)),
    )

    print(
        f"[{now()}] DPO start dataset={dataset} seed={args.seed} beta={args.beta} "
        f"pairs={len(valid_positions)} epochs={args.epochs}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_factor(
                epoch, args.warmup_epochs, args.epochs
            )

        losses, logits, implicit = [], [], []
        for batch_no, (x_w, x_l, batch_emb, batch_mask) in enumerate(pair_loader, 1):
            x_w = x_w.to(device)
            x_l = x_l.to(device)
            batch_emb = batch_emb.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)

            terms = []
            for _ in range(max(1, args.dpo_t_samples)):
                timestep = torch.randint(
                    t_low, t_high, (len(x_w),), device=device
                ).long()
                pair_noise = torch.randn_like(x_w)
                loss_w = diffusion.p_losses(
                    x_w, timestep, batch_emb, batch_mask,
                    noise=pair_noise, reduction="none",
                )
                loss_l = diffusion.p_losses(
                    x_l, timestep, batch_emb, batch_mask,
                    noise=pair_noise, reduction="none",
                )
                with torch.no_grad():
                    ref_w = ref_diffusion.p_losses(
                        x_w, timestep, batch_emb, batch_mask,
                        noise=pair_noise, reduction="none",
                    )
                    ref_l = ref_diffusion.p_losses(
                        x_l, timestep, batch_emb, batch_mask,
                        noise=pair_noise, reduction="none",
                    )
                terms.append(
                    -0.5 * args.beta * ((loss_w - loss_l) - (ref_w - ref_l))
                )

            logit = torch.stack(terms).mean(0)
            loss = -F.logsigmoid(logit).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DPO loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))
            logits.append(float(logit.abs().mean().item()))
            implicit.append(float((logit > 0).float().mean().item()))

            if (
                batch_no == 1
                or batch_no % args.log_interval_batches == 0
                or batch_no == len(pair_loader)
            ):
                print(
                    f"[{now()}] epoch {epoch+1}/{args.epochs} "
                    f"batch {batch_no}/{len(pair_loader)} "
                    f"loss={np.mean(losses):.4f} "
                    f"|logit|={np.mean(logits):.4f} "
                    f"implicit={np.mean(implicit):.3f}",
                    flush=True,
                )

        validation = None
        if epoch == 0 or (epoch + 1) % args.val_interval == 0:
            model.eval()
            validation = evaluate_validation(
                diffusion, val_loader, val_meta,
                args.validation_seed, args.val_ddim_steps,
                args.mse_temperature, args.selection_mse_weight, device,
            )
            threshold = best_score + args.early_stop_min_delta
            if validation["select_score"] > threshold:
                best_score = float(validation["select_score"])
                best_epoch = epoch + 1
                checks_without_improvement = 0
                torch.save(model.state_dict(), save_root / "best_model.pt")
                (save_root / "best_model_info.json").write_text(
                    json.dumps({
                        "selection_source": "validation",
                        "best_epoch": best_epoch,
                        "best_select_score": best_score,
                        "validation": validation,
                        "beta": args.beta,
                        "pair_source": "fixed_frozen_reference_cache",
                    }, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"[{now()}] new validation best epoch={best_epoch} "
                    f"select={best_score:.4f}",
                    flush=True,
                )
            else:
                checks_without_improvement += 1
                print(
                    f"[{now()}] no validation improvement "
                    f"{checks_without_improvement}/{args.early_stop_patience}; "
                    f"select={validation['select_score']:.4f} "
                    f"best={best_score:.4f}",
                    flush=True,
                )

        record = {
            "timestamp": now(),
            "dataset": dataset,
            "seed": args.seed,
            "epoch": epoch + 1,
            "beta": args.beta,
            "train": {
                "loss": float(np.mean(losses)),
                "inside_abs_mean": float(np.mean(logits)),
                "implicit_acc": float(np.mean(implicit)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
            "validation": validation,
        }
        with metric_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                model.state_dict(), save_root / f"ckpt_epoch{epoch+1}.pt"
            )

        if (
            epoch + 1 >= args.min_epochs
            and checks_without_improvement >= args.early_stop_patience
        ):
            print(f"[{now()}] early stopping at epoch {epoch+1}", flush=True)
            break

    torch.save(model.state_dict(), save_root / "final_model.pt")
    print(
        f"[{now()}] DPO complete best_epoch={best_epoch} "
        f"best_select={best_score:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
