#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_core.model import TimeSeriesDiffuserV2
from diffusion_core.scheduler import GaussianDiffusion
from reward_core.reward import TimeSeriesReward


class SourceDataset(Dataset):
    def __init__(self, x, emb, mask, indices):
        self.x = x
        self.emb = emb
        self.mask = mask
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = int(self.indices[position])
        return (
            torch.from_numpy(np.asarray(self.x[idx])).float(),
            torch.from_numpy(np.asarray(self.emb[idx])).float(),
            torch.from_numpy(np.asarray(self.mask[idx])).bool(),
            position,
            idx,
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args():
    p = argparse.ArgumentParser(description="Build fixed reward-ranked DPO pairs.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--pretrain_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pair_seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--valid_reward_eps", type=float, default=0.5)
    p.add_argument("--mse_temperature", type=float, default=0.5)
    p.add_argument("--hit_score", type=float, default=10.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset.lower()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    pair_seed = args.pair_seed if args.pair_seed is not None else 50000 + args.seed

    root = Path(args.data_dir) / dataset
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"

    required = (
        out / "winner.npy",
        out / "loser.npy",
        out / "reward_w.npy",
        out / "reward_l.npy",
        out / "reward_gap.npy",
        out / "valid_mask.npy",
        out / "train_indices.npy",
        manifest_path,
    )
    if not args.force and all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"Pair cache already complete: {out}", flush=True)
            return

    x = np.load(root / f"{dataset}_X.npy", mmap_mode="r")
    emb = np.load(root / f"{dataset}_emb.npy", mmap_mode="r")
    mask = np.load(root / f"{dataset}_emb_mask.npy", mmap_mode="r")
    train_indices = np.load(root / f"{dataset}_train_indices.npy")
    with (root / f"{dataset}_meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)

    checkpoint = Path(args.pretrain_dir) / dataset / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = TimeSeriesDiffuserV2(
        seq_len=96, text_dim=768, model_dim=256, num_layers=4, nhead=8
    ).to(device)
    model.load_state_dict(load_state(checkpoint, device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    diffusion = GaussianDiffusion(model, device=device)

    reward = TimeSeriesReward(
        device=device,
        dataset_name=dataset,
        reward_config={
            "use_trend": True,
            "use_volatility": True,
            "use_peak": True,
            "use_mse": True,
            "weight_trend": 1.0,
            "weight_volatility": 1.0,
            "weight_peak": 1.0,
            "weight_mse": 1.0,
            "hit_score": args.hit_score,
            "mse_temperature": args.mse_temperature,
            "use_relative_reward": False,
            "smooth_clip_enabled": False,
            "smooth_clip_scale": 15.0,
            "smooth_clip_max": 30.0,
        },
    )

    n = len(train_indices)
    shape = (n, 96, 1)
    winner = np.lib.format.open_memmap(
        out / "winner.npy", mode="w+", dtype=np.float16, shape=shape
    )
    loser = np.lib.format.open_memmap(
        out / "loser.npy", mode="w+", dtype=np.float16, shape=shape
    )
    reward_w = np.lib.format.open_memmap(
        out / "reward_w.npy", mode="w+", dtype=np.float32, shape=(n,)
    )
    reward_l = np.lib.format.open_memmap(
        out / "reward_l.npy", mode="w+", dtype=np.float32, shape=(n,)
    )
    reward_gap = np.lib.format.open_memmap(
        out / "reward_gap.npy", mode="w+", dtype=np.float32, shape=(n,)
    )
    valid_mask = np.lib.format.open_memmap(
        out / "valid_mask.npy", mode="w+", dtype=np.bool_, shape=(n,)
    )
    np.save(out / "train_indices.npy", train_indices)

    loader = DataLoader(
        SourceDataset(x, emb, mask, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(pair_seed)

    processed = 0
    with torch.no_grad():
        for batch_no, (batch_x, batch_emb, batch_mask, positions, raw_indices) in enumerate(loader, 1):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_size = batch_x.shape[0]
            batch_meta = [meta[int(idx)] for idx in raw_indices.tolist()]

            noise_a = torch.randn(
                (batch_size, 96, 1), generator=generator, device=device
            )
            noise_b = torch.randn(
                (batch_size, 96, 1), generator=generator, device=device
            )
            x_a = diffusion.ddim_sample(
                batch_emb, batch_mask, shape=(batch_size, 96, 1),
                ddim_steps=args.ddim_steps, eta=0.0, init_noise=noise_a,
            )
            x_b = diffusion.ddim_sample(
                batch_emb, batch_mask, shape=(batch_size, 96, 1),
                ddim_steps=args.ddim_steps, eta=0.0, init_noise=noise_b,
            )
            r_a, _ = reward.get_reward(
                x_a, batch_meta, target_series=batch_x, return_details=True
            )
            r_b, _ = reward.get_reward(
                x_b, batch_meta, target_series=batch_x, return_details=True
            )

            swap = r_b > r_a
            batch_w = torch.where(swap[:, None, None], x_b, x_a)
            batch_l = torch.where(swap[:, None, None], x_a, x_b)
            batch_rw = torch.where(swap, r_b, r_a)
            batch_rl = torch.where(swap, r_a, r_b)
            batch_gap = batch_rw - batch_rl

            pos = positions.numpy()
            winner[pos] = batch_w.detach().cpu().numpy().astype(np.float16)
            loser[pos] = batch_l.detach().cpu().numpy().astype(np.float16)
            reward_w[pos] = batch_rw.detach().cpu().numpy().astype(np.float32)
            reward_l[pos] = batch_rl.detach().cpu().numpy().astype(np.float32)
            reward_gap[pos] = batch_gap.detach().cpu().numpy().astype(np.float32)
            valid_mask[pos] = (
                batch_gap.detach().cpu().numpy() > args.valid_reward_eps
            )

            processed += batch_size
            if batch_no == 1 or batch_no % 20 == 0 or processed == n:
                print(
                    f"{dataset} seed={args.seed} pairs {processed}/{n} "
                    f"valid={float(valid_mask[:processed].mean()):.3f}",
                    flush=True,
                )

    for array in (winner, loser, reward_w, reward_l, reward_gap, valid_mask):
        array.flush()

    manifest = {
        "status": "complete",
        "dataset": dataset,
        "seed": args.seed,
        "pair_seed": pair_seed,
        "num_pairs": n,
        "num_valid_pairs": int(np.asarray(valid_mask).sum()),
        "valid_rate": float(np.asarray(valid_mask).mean()),
        "valid_reward_eps": args.valid_reward_eps,
        "ddim_steps": args.ddim_steps,
        "pair_source": "frozen_pretrained_reference",
        "ranking_reward": "equal_weight_trend_volatility_peak_fidelity",
        "component_wise_normalization": False,
        "pretrain_checkpoint": str(checkpoint),
        "pretrain_sha256": file_sha256(checkpoint),
        "storage_dtype": "float16",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
