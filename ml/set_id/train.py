"""Phased training loop with wandb logging.

Phase 1: freeze backbone, train heads only (Adam lr=1e-3).
Phase 2: unfreeze, cosine-schedule fine-tune (backbone lr=1e-5, heads lr=1e-4).

Metrics: per-attribute val accuracy + full-card val accuracy.
Best checkpoint is selected by mean per-attribute val accuracy.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from set_id.augment import train_transform, val_transform
from set_id.dataset import CardDataset, load_samples, stratified_split
from set_id.metrics import AccMeter
from set_id.model import build_model, param_count

ATTRS = ("number", "color", "shape", "shading")

# Per-arch defaults: ResNet-18 is fine-tuned from ImageNet (small LRs,
# phased training to protect pretrained weights). The custom CNN trains
# from scratch, so phase 1 (frozen random features) is pointless, and the
# whole net wants a normal scratch-training LR.
ARCH_DEFAULTS: dict[str, dict[str, float | int]] = {
    "resnet18": {"phase1_epochs": 10, "backbone_lr": 1e-5, "heads_lr": 1e-4},
    "small":    {"phase1_epochs": 0,  "backbone_lr": 1e-3, "heads_lr": 1e-3},
    "smaller":  {"phase1_epochs": 0,  "backbone_lr": 1e-3, "heads_lr": 1e-3},
}


@dataclass
class TrainConfig:
    arch: str = "resnet18"
    img_size: int = 128
    batch_size: int = 32
    phase1_epochs: int = 10
    phase2_epochs: int = 40
    phase1_lr: float = 1e-3
    backbone_lr: float = 1e-5
    heads_lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    holdout_identities: int = 15
    seed: int = 0
    amp: bool = True
    wandb_project: str | None = "set-classifier"
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"
    run_name: str | None = None
    crops_dir: Path = Path(__file__).resolve().parents[2] / "test-images" / "crops"
    out_dir: Path = Path(__file__).resolve().parents[1] / "checkpoints"
    # 021 is shot under orange light — pixel colors disagree with labels;
    # those crops poison training more than they help.
    exclude_sources: tuple[str, ...] = ("021",)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    samples = load_samples(cfg.crops_dir, exclude_sources=cfg.exclude_sources)
    train, val = stratified_split(samples, cfg.holdout_identities, cfg.seed)
    train_ds = CardDataset(train, train_transform(cfg.img_size))
    val_ds = CardDataset(val, val_transform(cfg.img_size))
    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    return train_dl, val_dl


def loss_fn(logits: tuple, labels: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = [F.cross_entropy(logits[i], labels[a]) for i, a in enumerate(ATTRS)]
    return sum(losses)


def unpack_batch(batch, device):
    img, number, color, shape, shading = batch
    return img.to(device, non_blocking=True), {
        "number": number.to(device, non_blocking=True),
        "color": color.to(device, non_blocking=True),
        "shape": shape.to(device, non_blocking=True),
        "shading": shading.to(device, non_blocking=True),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    amp: bool,
    *,
    desc: str = "train",
    step_offset: int = 0,
    lr_schedule=None,
) -> float:
    """Run one training epoch. If `lr_schedule` is given, it's called with the
    global step index (step_offset + local batch index) before each step, and
    is responsible for mutating `optimizer.param_groups[*]['lr']`.
    """
    model.train()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if lr_schedule is not None:
            lr_schedule(step_offset + i)
        img, labels = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(img)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * img.size(0)
        n += img.size(0)
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[AccMeter, float]:
    model.eval()
    meter = AccMeter()
    total_loss = 0.0
    n = 0
    for batch in loader:
        img, labels = unpack_batch(batch, device)
        logits = model(img)
        total_loss += loss_fn(logits, labels).item() * img.size(0)
        n += img.size(0)
        preds = {a: logits[i].argmax(1).tolist() for i, a in enumerate(ATTRS)}
        gts = {a: labels[a].tolist() for a in ATTRS}
        meter.update(preds, gts)
    return meter, total_loss / max(1, n)


def run(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if cfg.run_name is None:
        cfg.run_name = f"{cfg.arch}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    print(f"run: {cfg.run_name}")

    # wandb is optional; guard so missing/misconfigured wandb never blocks training
    wandb = None
    if cfg.wandb_mode != "disabled":
        try:
            import wandb as _wandb

            _wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                mode=cfg.wandb_mode,
                config=cfg.__dict__,
            )
            wandb = _wandb
        except Exception as e:
            print(f"wandb disabled ({e})")

    train_dl, val_dl = build_loaders(cfg)
    print(f"train samples: {len(train_dl.dataset)}  val samples: {len(val_dl.dataset)}")

    model = build_model(cfg.arch).to(device)
    print(f"arch: {cfg.arch}  params: {param_count(model):,}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_path = cfg.out_dir / f"{cfg.arch}_best.pt"

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    def end_of_epoch(phase: int, epoch_in_phase: int, tr_loss: float, extra_log: dict | None = None) -> None:
        nonlocal best_score
        meter, val_loss = evaluate(model, val_dl, device)
        m = meter.summary()
        score = m["acc/mean_attr"]
        global_epoch = epoch_in_phase if phase == 1 else cfg.phase1_epochs + epoch_in_phase
        log = {
            "phase": phase,
            "epoch": global_epoch,
            "train/loss": tr_loss,
            "val/loss": val_loss,
            **{f"val/{k}": v for k, v in m.items()},
        }
        if extra_log:
            log.update(extra_log)
        print(
            f"[p{phase} {epoch_in_phase:02d}] train={tr_loss:.4f} val={val_loss:.4f} "
            f"mean={score:.3f} full={m['acc/full_card']:.3f}"
        )
        if wandb:
            wandb.log(log)
        if score > best_score:
            best_score = score
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, best_path)

    # Phase 1: frozen backbone, train heads only
    model.freeze_backbone()
    phase1_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(phase1_params, lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.phase1_epochs):
        tr_loss = train_one_epoch(model, train_dl, opt, device, scaler, cfg.amp)
        end_of_epoch(1, epoch, tr_loss)

    # Phase 2: full fine-tune with cosine schedule
    model.unfreeze_backbone()
    opt = torch.optim.AdamW(
        [
            {"params": list(model.features.parameters()), "lr": cfg.backbone_lr},
            {"params": list(model.heads.parameters()), "lr": cfg.heads_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, cfg.phase2_epochs * len(train_dl))

    def cosine_schedule(global_step: int) -> None:
        mult = 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
        opt.param_groups[0]["lr"] = cfg.backbone_lr * mult
        opt.param_groups[1]["lr"] = cfg.heads_lr * mult

    step_offset = 0
    for epoch in range(cfg.phase2_epochs):
        tr_loss = train_one_epoch(
            model, train_dl, opt, device, scaler, cfg.amp,
            desc=f"p2/{epoch}",
            step_offset=step_offset,
            lr_schedule=cosine_schedule,
        )
        step_offset += len(train_dl)
        end_of_epoch(2, epoch, tr_loss, extra_log={
            "lr/backbone": opt.param_groups[0]["lr"],
            "lr/heads": opt.param_groups[1]["lr"],
        })

    print(f"best mean-attr val acc: {best_score:.4f}  → {best_path}")
    if wandb:
        wandb.summary["best/mean_attr"] = best_score
        wandb.finish()


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--phase1-epochs", type=int, default=None,
                   help="default: 10 for resnet18, 0 for small (scratch)")
    p.add_argument("--phase2-epochs", type=int, default=40)
    p.add_argument("--backbone-lr", type=float, default=None,
                   help="default: 1e-5 for resnet18, 1e-3 for small")
    p.add_argument("--heads-lr", type=float, default=None,
                   help="default: 1e-4 for resnet18, 1e-3 for small")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--holdout-identities", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--run-name", default=None)
    p.add_argument("--crops-dir", default=None)
    p.add_argument(
        "--exclude-sources",
        default=None,
        help="Comma-separated source image ids to drop (e.g. '021,024'). "
        "Pass empty string to include all. Default excludes '021' (orange-cast).",
    )
    args = p.parse_args()
    d = ARCH_DEFAULTS[args.arch]
    cfg = TrainConfig(
        arch=args.arch,
        img_size=args.img_size,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs if args.phase1_epochs is not None else d["phase1_epochs"],
        phase2_epochs=args.phase2_epochs,
        backbone_lr=args.backbone_lr if args.backbone_lr is not None else d["backbone_lr"],
        heads_lr=args.heads_lr if args.heads_lr is not None else d["heads_lr"],
        num_workers=args.num_workers,
        holdout_identities=args.holdout_identities,
        seed=args.seed,
        amp=not args.no_amp,
        wandb_mode=args.wandb_mode,
        run_name=args.run_name,
    )
    if args.crops_dir:
        cfg.crops_dir = Path(args.crops_dir)
    if args.exclude_sources is not None:
        cfg.exclude_sources = tuple(s for s in args.exclude_sources.split(",") if s)
    return cfg


if __name__ == "__main__":
    run(parse_args())
