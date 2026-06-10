import torch
import numpy as np
import os
import sys
import argparse
import random
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from diffusion_core.scheduler import GaussianDiffusion
from diffusion_core.model import TimeSeriesDiffuserV2

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def get_lr_schedule(epoch, warmup_epochs=50, total_epochs=800):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train(args):
    dataset = args.dataset.lower()
    data_dir = args.data_dir
    save_dir = os.path.join(args.save_dir, dataset)
    log_dir = args.log_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"diff_train_{dataset}.log")
    log_fh = open(log_path, 'w', buffering=1)

    def ts():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + '\n')
        log_fh.flush()

    log(f"[{ts()}] Diffusion pretraining started")
    log(f"[{ts()}] Dataset: {dataset}")
    log(f"[{ts()}] Device: {device}")
    log(f"[{ts()}] Config: 4L-256D, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"warmup={args.warmup_epochs}, patience={args.patience}, weight_decay=0.05, seed={args.seed}")

    all_x = np.load(os.path.join(data_dir, dataset, f"{dataset}_X.npy"))
    all_emb = np.load(os.path.join(data_dir, dataset, f"{dataset}_emb.npy"))
    all_emb_mask = np.load(os.path.join(data_dir, dataset, f"{dataset}_emb_mask.npy"))

    train_indices = np.load(os.path.join(data_dir, dataset, f"{dataset}_train_indices.npy"))
    n_train_all = len(train_indices)
    n_train = int(n_train_all * 0.9)
    train_idx = train_indices[:n_train]
    val_idx = train_indices[n_train + 95:]

    train_x = torch.from_numpy(all_x[train_idx]).float()
    train_emb = torch.from_numpy(all_emb[train_idx]).float()
    train_mask = torch.from_numpy(all_emb_mask[train_idx]).bool()

    val_x = torch.from_numpy(all_x[val_idx]).float()
    val_emb = torch.from_numpy(all_emb[val_idx]).float()
    val_mask = torch.from_numpy(all_emb_mask[val_idx]).bool()

    log(f"[{ts()}] Train samples: {len(train_x)} | Val samples: {len(val_x)}")
    log(f"[{ts()}] Data range: [{train_x.min():.4f}, {train_x.max():.4f}]")
    log(f"[{ts()}] Embedding shape: {train_emb.shape}")

    train_dataloader = DataLoader(
        TensorDataset(train_x, train_emb, train_mask),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device != "cpu"),
        drop_last=True,
    )
    val_dataloader = DataLoader(
        TensorDataset(val_x, val_emb, val_mask),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device != "cpu"),
    )

    model = TimeSeriesDiffuserV2(
        seq_len=96, text_dim=768, model_dim=256, num_layers=4, nhead=8
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[{ts()}] Parameters: {n_params:,}")

    diffusion = GaussianDiffusion(model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    log(f"[{ts()}] Training started")
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(args.epochs):
        lr_scale = get_lr_schedule(epoch, args.warmup_epochs, args.epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_scale

        model.train()
        epoch_train_loss = []
        for batch_x, batch_emb, batch_mask in train_dataloader:
            batch_x = batch_x.to(device)
            batch_emb = batch_emb.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (batch_x.shape[0],), device=device).long()
            loss = diffusion.p_losses(batch_x, t, batch_emb, batch_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss.append(loss.item())

        avg_train_loss = np.mean(epoch_train_loss)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for batch_x, batch_emb, batch_mask in val_dataloader:
                batch_x = batch_x.to(device)
                batch_emb = batch_emb.to(device)
                batch_mask = batch_mask.to(device)
                t = torch.randint(0, diffusion.timesteps, (batch_x.shape[0],), device=device).long()
                loss = diffusion.p_losses(batch_x, t, batch_emb, batch_mask)
                epoch_val_loss.append(loss.item())

        avg_val_loss = np.mean(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            msg = (f"[{ts()}] Epoch {epoch+1}/{args.epochs} | "
                   f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                   f"EarlyStop: {early_stop_counter}/{args.patience}")
            log(msg)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            if (epoch + 1) % 10 == 0:
                log(f"[{ts()}] Saved best model (val_loss={best_val_loss:.6f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                log(f"[{ts()}] Early stopping triggered (no improvement for {args.patience} epochs, epoch={epoch+1})")
                break

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            log(f"[{ts()}] Saved checkpoint epoch {epoch+1}")

    log(f"[{ts()}] Pretraining finished. Best val loss: {best_val_loss:.6f}")
    log(f"[{ts()}] Best model: {save_dir}/best_model.pt")
    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion model pretraining (4L-256D)')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--save_dir', type=str, default='checkpoints/diff_train')
    parser.add_argument('--log_dir', type=str, default='logs/diff_train')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)
