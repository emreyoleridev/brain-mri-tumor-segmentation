"""Train the U-Net on the train split, select the checkpoint with the best validation Dice."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bmts.data import MRISegDataset  # noqa: E402
from bmts.metrics import bce_dice_loss  # noqa: E402
from bmts.model import UNet, pick_device  # noqa: E402


@torch.no_grad()
def validate(model, loader, device):
    """Mean loss and dataset-level (global) Dice / IoU at threshold 0.5."""
    model.eval()
    losses, inter, psum, tsum, union = [], 0.0, 0.0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(bce_dice_loss(logits, y).item())
        p = (torch.sigmoid(logits) > 0.5).float()
        inter += (p * y).sum().item()
        psum += p.sum().item()
        tsum += y.sum().item()
        union += ((p + y) > 0).sum().item()
    return float(np.mean(losses)), 2 * inter / (psum + tsum + 1e-6), inter / (union + 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="U-Net width (channels in first block)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tumor-oversample", type=float, default=2.0,
                    help="sampling weight of tumor slices relative to empty slices")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", type=Path, default=None, help="checkpoint to fine-tune from (continues its history)")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "unet_best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    df = pd.read_csv(ROOT / "data" / "index.csv")
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]

    # tumor slices are ~35 % of the data: oversample them so each batch carries enough foreground
    weights = np.where(train_df["has_tumor"], args.tumor_oversample, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)
    train_dl = DataLoader(MRISegDataset(train_df, args.size, train=True, seed=args.seed), batch_size=args.batch,
                          sampler=sampler, num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    val_dl = DataLoader(MRISegDataset(val_df, args.size), batch_size=32, num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    model = UNet(base=args.base).to(device)
    history, best, start = [], -1.0, 0
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start, best = ckpt["meta"]["epoch"], ckpt["meta"]["val_dice"]
        hist_csv = ROOT / "results" / "training_history.csv"
        if hist_csv.exists():
            history = pd.read_csv(hist_csv).to_dict("records")
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    print(f"device={device} params={n_params / 1e6:.2f}M train={len(train_df)} val={len(val_df)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start + 1, start + args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = bce_dice_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val_loss, val_dice, val_iou = validate(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_dice": val_dice,
               "val_iou": val_iou, "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
        history.append(rec)
        flag = ""
        if val_dice > best:
            best = val_dice
            torch.save({"config": model.config, "state_dict": model.state_dict(),
                        "meta": {"epoch": epoch, "val_dice": val_dice, "val_iou": val_iou, "params": n_params,
                                 "args": {k: str(v) for k, v in vars(args).items()}}}, args.out)
            flag = "  *best*"
        print(f"[{epoch:02d}/{start + args.epochs}] train {rec['train_loss']:.4f} | val {val_loss:.4f} "
              f"dice {val_dice:.4f} iou {val_iou:.4f} | {rec['sec']:.0f}s{flag}", flush=True)
        pd.DataFrame(history).to_csv(ROOT / "results" / "training_history.csv", index=False)

    (ROOT / "results" / "training_summary.json").write_text(json.dumps(
        {"best_val_dice": best, "best_epoch": int(pd.DataFrame(history).set_index("epoch").val_dice.idxmax()), "params": n_params,
         "device": str(device), "total_min": sum(h["sec"] for h in history) / 60, **vars(args)}, indent=2, default=str))


if __name__ == "__main__":
    main()
