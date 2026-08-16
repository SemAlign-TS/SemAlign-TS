import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion_core.model import TimeSeriesDiffuserV2
from diffusion_core.scheduler import GaussianDiffusion
from reward_core.reward import TimeSeriesReward, classify_trend, calc_trend_soft_score

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

device = "cuda" if torch.cuda.is_available() else "cpu"

REWARD_COMPONENTS = ["trend", "volatility", "peak", "mse"]

REWARD_MODE_COMPONENTS = {
    "full": ["trend", "volatility", "peak", "mse"],
    "mse_only": ["mse"],
    "no_trend": ["volatility", "peak", "mse"],
    "no_volatility": ["trend", "peak", "mse"],
    "no_peak": ["trend", "volatility", "mse"],
}


def get_active_reward_components(reward_mode):
    if reward_mode not in REWARD_MODE_COMPONENTS:
        raise ValueError(
            f"Unknown reward_mode={reward_mode}. "
            f"Available: {sorted(REWARD_MODE_COMPONENTS)}"
        )
    return REWARD_MODE_COMPONENTS[reward_mode]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_schedule(epoch, warmup_epochs=10, total_epochs=40):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs

    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))


def grpo_advantage(rewards, eps=1e-6, clip_value=2.5):
    mean_r = rewards.mean(dim=0, keepdim=True)
    std_r = rewards.std(dim=0, unbiased=False, keepdim=True)
    advantage = (rewards - mean_r) / (std_r + eps)
    return torch.clamp(advantage, -clip_value, clip_value)


def combine_component_rewards(
    component_stacks,
    reward_mode="full",
):
    active_components = get_active_reward_components(reward_mode)

    first_key = active_components[0]
    total = torch.zeros_like(component_stacks[first_key])
    component_debug = {}

    for name in active_components:
        raw_component = component_stacks[name]

        weight = 1.0
        total = total + weight * raw_component

        component_debug[f"{name}_raw_mean"] = float(raw_component.mean().item())
        component_debug[f"{name}_raw_std"] = float(
            raw_component.std(unbiased=False).item()
        )
        component_debug[f"{name}_weight"] = weight

    return total, component_debug


def classify_volatility(volatility):
    if volatility < 0.5:
        return "low"
    if volatility < 0.8:
        return "medium"
    return "high"


def classify_peak(peak):
    if peak < 1 / 3:
        return "early"
    if peak < 2 / 3:
        return "middle"
    return "late"


def extract_series_features(series):
    batch_size, seq_len, _ = series.shape
    series_2d = series.squeeze(-1)

    x_axis = (
        torch.arange(seq_len, device=series.device)
        .float()
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    x_mean = x_axis.mean(dim=1, keepdim=True)
    y_mean = series_2d.mean(dim=1, keepdim=True)

    numerator = ((x_axis - x_mean) * (series_2d - y_mean)).sum(dim=1)
    denominator = ((x_axis - x_mean) ** 2).sum(dim=1)
    slope = numerator / (denominator + 1e-8)

    value_range = series_2d.max(dim=1)[0] - series_2d.min(dim=1)[0]
    normalized_slope = slope * seq_len / (value_range + 1e-8)

    trend_line = slope.unsqueeze(1) * x_axis + (
        y_mean - slope.unsqueeze(1) * x_mean
    )
    detrended = series_2d - trend_line
    normalized_volatility = detrended.std(dim=1) / (series_2d.std(dim=1) + 1e-8)
    peak_location = series_2d.argmax(dim=1).float() / seq_len

    return normalized_slope, normalized_volatility, peak_location


def calculate_alignment_metrics(series, target_meta, target_series, mse_temperature):
    slope, volatility, peak = extract_series_features(series)

    trend_hits = []
    vol_hits = []
    peak_hits = []
    trend_soft_scores = []

    for i, meta in enumerate(target_meta):
        target_trend = meta.get("trend", "")
        slope_value = slope[i].item()

        trend_hits.append(1.0 if classify_trend(slope_value) == target_trend else 0.0)
        trend_soft_scores.append(calc_trend_soft_score(slope_value, target_trend))

        vol_hits.append(
            1.0
            if classify_volatility(volatility[i].item()) == meta.get("volatility", "")
            else 0.0
        )
        peak_hits.append(
            1.0
            if classify_peak(peak[i].item()) == meta.get("peak_location", "")
            else 0.0
        )

    mse = F.mse_loss(series, target_series, reduction="none").mean(dim=(1, 2))
    mse_score = torch.exp(-mse / max(mse_temperature, 1e-6))

    return {
        "trend_acc": float(np.mean(trend_hits)),
        "trend_soft_score": float(np.mean(trend_soft_scores)),
        "volatility_acc": float(np.mean(vol_hits)),
        "peak_acc": float(np.mean(peak_hits)),
        "mse_score": float(mse_score.mean().item()),
    }


@torch.no_grad()
def evaluate_validation(
    diffusion,
    val_loader,
    val_meta,
    validation_seed,
    ddim_steps,
    mse_temperature,
    selection_mse_weight,
):
    """Evaluate one deterministic DDIM sample per validation condition."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(validation_seed))

    totals = {
        "trend_acc": 0.0,
        "trend_soft_score": 0.0,
        "volatility_acc": 0.0,
        "peak_acc": 0.0,
        "mse_score": 0.0,
    }
    total_n = 0

    for batch_x, batch_emb, batch_mask, batch_idx in val_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_emb = batch_emb.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)
        batch_meta = [val_meta[int(i)] for i in batch_idx.tolist()]
        batch_size = int(batch_x.shape[0])

        init_noise = torch.randn(
            (batch_size, 96, 1),
            generator=generator,
            device=device,
        )
        generated = diffusion.ddim_sample(
            batch_emb,
            batch_mask,
            shape=(batch_size, 96, 1),
            ddim_steps=ddim_steps,
            eta=0.0,
            init_noise=init_noise,
        )
        metrics = calculate_alignment_metrics(
            generated,
            batch_meta,
            batch_x,
            mse_temperature,
        )
        for key in totals:
            totals[key] += float(metrics[key]) * batch_size
        total_n += batch_size

    if total_n <= 0:
        raise RuntimeError("Validation split is empty")

    result = {key: value / total_n for key, value in totals.items()}
    result["semantic_score"] = (
        result["trend_acc"]
        + result["volatility_acc"]
        + result["peak_acc"]
    ) / 3.0
    result["select_score"] = (
        result["semantic_score"]
        + float(selection_mse_weight) * result["mse_score"]
    )
    result["num_samples"] = total_n
    return result


def assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite tensor detected: {name}")


def train_osra(args):
    set_seed(args.seed)

    if args.group_size < 2:
        raise ValueError("OSRA requires --group_size >= 2.")

    dataset = args.dataset.lower()
    experiment_name = args.experiment_name

    save_dir = os.path.join(args.save_dir, experiment_name, dataset)
    log_dir = os.path.join(args.log_dir, experiment_name)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    metric_log_path = os.path.join(log_dir, f"{dataset}.jsonl")
    config_path = os.path.join(save_dir, "run_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    pretrain_dir = os.path.join(args.pretrain_dir, dataset)
    pretrain_path = os.path.join(pretrain_dir, "best_model.pt")

    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrain_path}. "
            f"Please check --pretrain_dir or run pretraining first."
        )

    t_low_ratio = max(0.0, min(1.0, args.t_low_ratio))
    t_high_ratio = max(0.0, min(1.0, args.t_high_ratio))

    if t_high_ratio <= t_low_ratio:
        t_high_ratio = min(1.0, t_low_ratio + 0.01)

    def ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{ts()}] OSRA Post-training Start", flush=True)
    print(
        f"[{ts()}] exp={experiment_name} dataset={dataset} "
        f"epochs={args.epochs} batch={args.batch_size} "
        f"G={args.group_size} ddim={args.ddim_steps} "
        f"kl_coef={args.kl_coef} lr={args.lr} "
        f"warmup={args.warmup_epochs} "
        f"t_range=[{t_low_ratio:.2f},{t_high_ratio:.2f}] "
        f"reward_mode={args.reward_mode} seed={args.seed}",
        flush=True,
    )
    print(f"[{ts()}] Save dir: {save_dir}", flush=True)
    print(f"[{ts()}] Metric log: {metric_log_path}", flush=True)
    print(f"[{ts()}] Run config: {config_path}", flush=True)

    all_x = np.load(os.path.join(args.data_dir, dataset, f"{dataset}_X.npy"))
    all_emb = np.load(os.path.join(args.data_dir, dataset, f"{dataset}_emb.npy"))
    all_emb_mask = np.load(
        os.path.join(args.data_dir, dataset, f"{dataset}_emb_mask.npy")
    )

    with open(os.path.join(args.data_dir, dataset, f"{dataset}_meta.pkl"), "rb") as fh:
        all_meta = pickle.load(fh)

    train_indices = np.load(
        os.path.join(args.data_dir, dataset, f"{dataset}_train_indices.npy")
    )
    val_indices = np.load(
        os.path.join(args.data_dir, dataset, f"{dataset}_val_indices.npy")
    )

    train_x = torch.from_numpy(all_x[train_indices]).float()
    train_emb = torch.from_numpy(all_emb[train_indices]).float()
    train_mask = torch.from_numpy(all_emb_mask[train_indices]).bool()
    train_meta = [all_meta[i] for i in train_indices]

    if args.val_max_samples > 0 and len(val_indices) > args.val_max_samples:
        rng = np.random.RandomState(args.validation_seed)
        val_indices = np.sort(
            rng.choice(val_indices, size=args.val_max_samples, replace=False)
        )
    val_x = torch.from_numpy(all_x[val_indices]).float()
    val_emb = torch.from_numpy(all_emb[val_indices]).float()
    val_mask = torch.from_numpy(all_emb_mask[val_indices]).bool()
    val_meta = [all_meta[i] for i in val_indices]

    print(
        f"[{ts()}] Train samples: {len(train_x)} | Validation samples: {len(val_x)}",
        flush=True,
    )

    dataloader = DataLoader(
        TensorDataset(train_x, train_emb, train_mask, torch.arange(len(train_x))),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_emb, val_mask, torch.arange(len(val_x))),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = TimeSeriesDiffuserV2(
        seq_len=96,
        text_dim=768,
        model_dim=256,
        num_layers=4,
        nhead=8,
    ).to(device)

    state_dict = torch.load(pretrain_path, map_location=device)
    model.load_state_dict(state_dict)

    print(f"[{ts()}] Loaded pretrain policy: {pretrain_path}", flush=True)

    ref_model = TimeSeriesDiffuserV2(
        seq_len=96,
        text_dim=768,
        model_dim=256,
        num_layers=4,
        nhead=8,
    ).to(device)

    ref_model.load_state_dict(state_dict)
    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(
        f"[{ts()}] Ref frozen | Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )

    diffusion = GaussianDiffusion(model, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    t_low = int(t_low_ratio * diffusion.timesteps)
    t_high = min(
        max(t_low + 1, int(t_high_ratio * diffusion.timesteps)),
        diffusion.timesteps,
    )

    G = args.group_size
    best_select_score = -float("inf")
    best_epoch = -1
    validation_checks_without_improvement = 0

    clip_eps = args.clip_eps
    grad_clip = args.grad_clip
    for epoch in range(args.epochs):
        phase_name = "StaticSemanticWeights"
        cur_weights = {
            "weight_trend": 1.0,
            "weight_volatility": 1.0,
            "weight_peak": 1.0,
            "weight_mse": 1.0,
        }

        active_reward_components = get_active_reward_components(args.reward_mode)
        use_trend_reward = "trend" in active_reward_components
        use_volatility_reward = "volatility" in active_reward_components
        use_peak_reward = "peak" in active_reward_components
        use_mse_reward = "mse" in active_reward_components

        reward_calc = TimeSeriesReward(
            device=device,
            dataset_name=dataset,
            reward_config={
                "use_trend": use_trend_reward,
                "use_volatility": use_volatility_reward,
                "use_peak": use_peak_reward,
                "use_mse": use_mse_reward,
                "weight_trend": 1.0,
                "weight_volatility": 1.0,
                "weight_peak": 1.0,
                "weight_mse": 1.0,
                "hit_score": args.hit_score,
                "mse_temperature": args.mse_temperature,
                "use_relative_reward": False,
                "smooth_clip_enabled": False,
                "smooth_clip_scale": args.smooth_clip_scale,
                "smooth_clip_max": args.smooth_clip_max,
            },
        )

        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * get_lr_schedule(
                epoch,
                args.warmup_epochs,
                args.epochs,
            )

        ep_loss = []
        ep_pg_loss = []
        ep_kl = []

        ep_r_mean = []
        ep_r_std = []
        ep_best_r = []
        ep_worst_r = []

        ep_trend = []
        ep_vol = []
        ep_peak = []
        ep_mse_r = []

        ep_trend_acc = []
        ep_trend_soft = []
        ep_vol_acc = []
        ep_peak_acc = []
        ep_mse_score = []

        ep_component_debug = []

        for batch_no, (batch_x, batch_emb, batch_mask, batch_idx) in enumerate(dataloader, start=1):
            batch_x = batch_x.to(device)
            batch_emb = batch_emb.to(device)
            batch_mask = batch_mask.to(device)

            B = batch_emb.shape[0]
            batch_meta = [train_meta[i.item()] for i in batch_idx]

            optimizer.zero_grad()

            with torch.no_grad():
                all_samples = []
                component_buffers = {name: [] for name in REWARD_COMPONENTS}
                batch_metric_list = []

                for g in range(G):
                    noise_g = torch.randn(B, 96, 1, device=device)

                    x_g = diffusion.ddim_sample(
                        batch_emb,
                        batch_mask,
                        shape=(B, 96, 1),
                        ddim_steps=args.ddim_steps,
                        eta=0.0,
                        cfg_scale=1.0,
                        init_noise=noise_g,
                    )

                    _, info_g = reward_calc.get_raw_reward(
                        x_g,
                        batch_meta,
                        target_series=batch_x,
                    )

                    all_samples.append(x_g)

                    zero_component = torch.zeros(B, device=device)

                    for name in REWARD_COMPONENTS:
                        component_buffers[name].append(
                            info_g.get(name, zero_component).detach()
                        )

                    alignment_metrics = calculate_alignment_metrics(
                        x_g,
                        batch_meta,
                        batch_x,
                        args.mse_temperature,
                    )
                    batch_metric_list.append(alignment_metrics)

                component_stacks = {
                    name: torch.stack(component_buffers[name], dim=0)
                    for name in REWARD_COMPONENTS
                }

                rewards_tensor, component_debug = combine_component_rewards(
                    component_stacks,
                    reward_mode=args.reward_mode,
                )

                assert_finite_tensor("rewards_tensor", rewards_tensor)

                advantage = grpo_advantage(
                    rewards_tensor,
                    clip_value=args.adv_clip,
                )

                assert_finite_tensor("advantage", advantage)

                ep_r_mean.append(rewards_tensor.mean().item())
                ep_r_std.append(rewards_tensor.std(unbiased=False).item())
                ep_best_r.append(rewards_tensor.max(dim=0).values.mean().item())
                ep_worst_r.append(rewards_tensor.min(dim=0).values.mean().item())

                ep_trend.append(component_stacks["trend"].mean().item())
                ep_vol.append(component_stacks["volatility"].mean().item())
                ep_peak.append(component_stacks["peak"].mean().item())
                ep_mse_r.append(component_stacks["mse"].mean().item())

                avg_metrics = {
                    key: float(np.mean([m[key] for m in batch_metric_list]))
                    for key in batch_metric_list[0].keys()
                }

                ep_trend_acc.append(avg_metrics["trend_acc"])
                ep_trend_soft.append(avg_metrics.get("trend_soft_score", 0.0))
                ep_vol_acc.append(avg_metrics["volatility_acc"])
                ep_peak_acc.append(avg_metrics["peak_acc"])
                ep_mse_score.append(avg_metrics["mse_score"])

                ep_component_debug.append(component_debug)

            pg_loss_total = torch.tensor(0.0, device=device)
            kl_loss_total = torch.tensor(0.0, device=device)

            model.train()

            for g in range(G):
                x_g = all_samples[g]
                adv_g = advantage[g].detach()

                t_k = torch.randint(t_low, t_high, (B,), device=device).long()
                noise = torch.randn_like(x_g)

                x_t = diffusion.q_sample(x_g, t_k, noise=noise)

                pred_noise_theta = model(x_t, t_k, batch_emb, batch_mask)

                loss_theta = F.mse_loss(
                    pred_noise_theta,
                    noise,
                    reduction="none",
                ).mean(dim=[1, 2])

                with torch.no_grad():
                    pred_noise_ref = ref_model(x_t, t_k, batch_emb, batch_mask)

                    loss_ref = F.mse_loss(
                        pred_noise_ref,
                        noise,
                        reduction="none",
                    ).mean(dim=[1, 2])

                ratio = torch.exp(
                    torch.clamp(
                        loss_ref.detach() - loss_theta,
                        min=-args.ratio_clip,
                        max=args.ratio_clip,
                    )
                )

                surr1 = ratio * adv_g
                surr2 = torch.clamp(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                ) * adv_g

                pg_loss = -torch.min(surr1, surr2).mean()

                kl_loss = F.mse_loss(
                    pred_noise_theta,
                    pred_noise_ref.detach(),
                    reduction="mean",
                )

                pg_loss_total = pg_loss_total + pg_loss
                kl_loss_total = kl_loss_total + kl_loss

            pg_loss_total = pg_loss_total / G
            kl_loss_total = kl_loss_total / G

            loss = pg_loss_total + args.kl_coef * kl_loss_total

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch {epoch + 1}: "
                    f"loss={loss.item()}, pg={pg_loss_total.item()}, "
                    f"kl={kl_loss_total.item()}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite grad norm detected: {grad_norm}"
                )

            optimizer.step()

            ep_loss.append(loss.item())
            ep_pg_loss.append(pg_loss_total.item())
            ep_kl.append(kl_loss_total.item())

            if (
                args.log_interval_batches > 0
                and (
                    batch_no == 1
                    or batch_no % args.log_interval_batches == 0
                    or batch_no == len(dataloader)
                )
            ):
                print(
                    f"[{ts()}] Epoch {epoch + 1}/{args.epochs} "
                    f"batch {batch_no}/{len(dataloader)} | "
                    f"Loss:{float(np.mean(ep_loss)):.4f} "
                    f"PG:{float(np.mean(ep_pg_loss)):.4f} "
                    f"KL:{float(np.mean(ep_kl)):.4f} "
                    f"R:{float(np.mean(ep_r_mean)):.3f}",
                    flush=True,
                )

        al = float(np.mean(ep_loss))
        apg = float(np.mean(ep_pg_loss))
        akl = float(np.mean(ep_kl))

        arm = float(np.mean(ep_r_mean))
        ars = float(np.mean(ep_r_std))
        abr = float(np.mean(ep_best_r))
        awr = float(np.mean(ep_worst_r))

        at = float(np.mean(ep_trend)) if ep_trend else 0.0
        avo = float(np.mean(ep_vol)) if ep_vol else 0.0
        ap = float(np.mean(ep_peak)) if ep_peak else 0.0
        am = float(np.mean(ep_mse_r)) if ep_mse_r else 0.0

        trend_acc = float(np.mean(ep_trend_acc)) if ep_trend_acc else 0.0
        trend_soft = float(np.mean(ep_trend_soft)) if ep_trend_soft else 0.0
        vol_acc = float(np.mean(ep_vol_acc)) if ep_vol_acc else 0.0
        peak_acc = float(np.mean(ep_peak_acc)) if ep_peak_acc else 0.0
        mse_score = float(np.mean(ep_mse_score)) if ep_mse_score else 0.0

        semantic_score = (trend_acc + vol_acc + peak_acc) / 3.0
        train_select_score = semantic_score + args.selection_mse_weight * mse_score

        run_validation = (epoch == 0) or ((epoch + 1) % args.val_interval == 0)
        val_metrics = None
        if run_validation:
            val_metrics = evaluate_validation(
                diffusion=diffusion,
                val_loader=val_loader,
                val_meta=val_meta,
                validation_seed=args.validation_seed,
                ddim_steps=args.val_ddim_steps,
                mse_temperature=args.mse_temperature,
                selection_mse_weight=args.selection_mse_weight,
            )

        lr_now = float(optimizer.param_groups[0]["lr"])
        cur_ts = ts()

        component_debug_mean = {}
        if ep_component_debug:
            keys = ep_component_debug[0].keys()
            for key in keys:
                component_debug_mean[key] = float(
                    np.mean([d[key] for d in ep_component_debug if key in d])
                )

        epoch_log = {
            "timestamp": cur_ts,
            "experiment": experiment_name,
            "dataset": dataset,
            "epoch": epoch + 1,
            "phase": phase_name,
            "reward_mode": args.reward_mode,
            "seed": args.seed,
            "loss": al,
            "pg_loss": apg,
            "kl_loss": akl,
            "reward_mean": arm,
            "reward_std": ars,
            "best_reward": abr,
            "worst_reward": awr,
            "trend_reward_raw": at,
            "vol_reward_raw": avo,
            "peak_reward_raw": ap,
            "mse_reward_raw": am,
            "trend_acc": trend_acc,
            "trend_soft_score": trend_soft,
            "vol_acc": vol_acc,
            "peak_acc": peak_acc,
            "mse_score": mse_score,
            "semantic_score": semantic_score,
            "train_select_score": train_select_score,
            "validation": val_metrics,
            "weight_trend": float(cur_weights["weight_trend"]),
            "weight_volatility": float(cur_weights["weight_volatility"]),
            "weight_peak": float(cur_weights["weight_peak"]),
            "weight_mse": float(cur_weights["weight_mse"]),
            "lr": lr_now,
            **component_debug_mean,
        }

        with open(metric_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_log, ensure_ascii=False) + "\n")

        print(
            f"[{cur_ts}] Epoch {epoch + 1}/{args.epochs} [{phase_name}] | "
            f"Loss:{al:.4f} | PG:{apg:.4f} | KL:{akl:.4f} | "
            f"R_mean:{arm:.3f} R_std:{ars:.3f} | "
            f"Best_R:{abr:.3f} Worst_R:{awr:.3f} | "
            f"RawReward T/V/P/M:{at:.3f}/{avo:.3f}/{ap:.3f}/{am:.3f} | "
            f"Acc T/V/P:{trend_acc:.3f}/{vol_acc:.3f}/{peak_acc:.3f} "
            f"Semantic:{semantic_score:.3f} "
            f"TrendSoft:{trend_soft:.3f} "
            f"MSEScore:{mse_score:.3f} "
            f"TrainSelect:{train_select_score:.3f} | "
            f"W(T/V/P/M):{cur_weights['weight_trend']:.2f}/"
            f"{cur_weights['weight_volatility']:.2f}/"
            f"{cur_weights['weight_peak']:.2f}/"
            f"{cur_weights['weight_mse']:.2f} | "
            f"LR:{lr_now:.6f}",
            flush=True,
        )

        if val_metrics is not None:
            print(
                f"[{cur_ts}] Validation | n={val_metrics['num_samples']} | "
                f"Acc T/V/P:{val_metrics['trend_acc']:.3f}/"
                f"{val_metrics['volatility_acc']:.3f}/"
                f"{val_metrics['peak_acc']:.3f} | "
                f"Semantic:{val_metrics['semantic_score']:.3f} | "
                f"MSEScore:{val_metrics['mse_score']:.3f} | "
                f"Select:{val_metrics['select_score']:.3f}",
                flush=True,
            )

            improvement_threshold = (
                best_select_score + float(args.early_stop_min_delta)
            )
            if val_metrics["select_score"] > improvement_threshold:
                best_select_score = float(val_metrics["select_score"])
                best_epoch = epoch + 1
                validation_checks_without_improvement = 0

                best_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)

                best_info_path = os.path.join(save_dir, "best_model_info.json")
                with open(best_info_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "selection_source": "validation",
                            "best_epoch": best_epoch,
                            "best_select_score": best_select_score,
                            "validation": val_metrics,
                            "train_metrics_at_best_epoch": {
                                "semantic_score": semantic_score,
                                "mse_score": mse_score,
                                "train_select_score": train_select_score,
                            },
                            "early_stopping": {
                                "min_delta": float(args.early_stop_min_delta),
                                "patience_validation_checks": int(
                                    args.early_stop_patience
                                ),
                                "min_epochs": int(args.min_epochs),
                            },
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                print(
                    f"[{cur_ts}] New validation-best model -> {best_path} | "
                    f"Select:{best_select_score:.4f} "
                    f"Semantic:{val_metrics['semantic_score']:.4f} "
                    f"MSEScore:{val_metrics['mse_score']:.4f}",
                    flush=True,
                )
            else:
                validation_checks_without_improvement += 1
                print(
                    f"[{cur_ts}] No validation improvement | "
                    f"checks={validation_checks_without_improvement}/"
                    f"{args.early_stop_patience} | "
                    f"best={best_select_score:.4f}",
                    flush=True,
                )

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{cur_ts}] Checkpoint -> {ckpt_path}", flush=True)

        if (
            args.early_stop_patience > 0
            and (epoch + 1) >= args.min_epochs
            and validation_checks_without_improvement
            >= args.early_stop_patience
        ):
            print(
                f"[{ts()}] Early stopping at epoch {epoch + 1}: "
                f"no validation improvement larger than "
                f"{args.early_stop_min_delta:g} for "
                f"{validation_checks_without_improvement} validation checks.",
                flush=True,
            )
            break

    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)

    print(f"[{ts()}] Final model saved -> {final_path}", flush=True)
    print(
        f"[{ts()}] Done. Best Select Score: {best_select_score:.4f} "
        f"at epoch {best_epoch}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="osra")
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--log_dir", type=str, default="outputs/logs/osra")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--pretrain_dir", type=str, default="outputs/checkpoints/pretrain")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--adv_clip", type=float, default=2.5)
    parser.add_argument("--ratio_clip", type=float, default=5.0)

    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--t_low_ratio", type=float, default=0.1)
    parser.add_argument("--t_high_ratio", type=float, default=0.9)

    parser.add_argument(
        "--reward_mode",
        type=str,
        default="full",
        choices=["full", "mse_only", "no_trend", "no_volatility", "no_peak"],
    )

    parser.add_argument("--selection_mse_weight", type=float, default=0.2)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--val_ddim_steps", type=int, default=20)
    parser.add_argument("--val_max_samples", type=int, default=0)
    parser.add_argument("--validation_seed", type=int, default=91021)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many validation checks without improvement; 0 disables.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation Select improvement required to reset patience.",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=1,
        help="Minimum number of epochs before early stopping is allowed.",
    )
    parser.add_argument(
        "--log_interval_batches",
        type=int,
        default=50,
        help="Print live batch progress every N batches; 0 disables.",
    )

    parser.add_argument("--hit_score", type=float, default=10.0)
    parser.add_argument("--mse_temperature", type=float, default=0.5)
    parser.add_argument("--smooth_clip_scale", type=float, default=15.0)
    parser.add_argument("--smooth_clip_max", type=float, default=30.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_osra(args)