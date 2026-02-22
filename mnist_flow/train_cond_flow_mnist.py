import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mnist_flow.models.cond_flow import ConditionalRealNVP, preprocess
from mnist_flow.utils.checkpoint import ExponentialMovingAverage, save_checkpoint


@dataclass
class Config:
    data_dir: str
    out_dir: str
    batch_size: int
    epochs: int
    lr: float
    num_flows: int
    hidden_dim: int
    label_emb_dim: int
    num_workers: int
    val_split: int
    amp: bool
    compile_model: bool
    sample_every: int
    ema_decay: float
    save_best: bool
    seed: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Conditional RealNVP for MNIST")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./outputs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-flows", type=int, default=12)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--label-emb-dim", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-split", type=int, default=5000)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true", dest="compile_model")
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save best.pt by val bpd",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Config(
        data_dir=a.data_dir,
        out_dir=a.out_dir,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        num_flows=a.num_flows,
        hidden_dim=a.hidden_dim,
        label_emb_dim=a.label_emb_dim,
        num_workers=a.num_workers,
        val_split=a.val_split,
        amp=a.amp,
        compile_model=a.compile_model,
        sample_every=a.sample_every,
        ema_decay=a.ema_decay,
        save_best=a.save_best,
        seed=a.seed,
    )


def evaluate(model: ConditionalRealNVP, loader: DataLoader, device: torch.device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            y, log_det_pre = preprocess(x)
            y = y.flatten(1)
            log_prob = model.log_prob(y, labels) + log_det_pre
            nll = -log_prob.mean()
            running_loss += nll.item()
    avg_nll = running_loss / max(1, len(loader))
    bpd = avg_nll / (28 * 28 * math.log(2.0))
    return avg_nll, bpd


def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    print(f"device={device}")

    full_train_set = datasets.MNIST(
        cfg.data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    if not (1 <= cfg.val_split < len(full_train_set)):
        raise ValueError(f"val_split must be in [1, {len(full_train_set)-1}]")
    train_len = len(full_train_set) - cfg.val_split
    split_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(
        full_train_set, [train_len, cfg.val_split], generator=split_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )

    model = ConditionalRealNVP(
        dim=28 * 28,
        num_flows=cfg.num_flows,
        hidden=cfg.hidden_dim,
        label_emb_dim=cfg.label_emb_dim,
    ).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ExponentialMovingAverage(model, decay=cfg.ema_decay)
    best_val_bpd = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            y, log_det_pre = preprocess(x)
            y = y.flatten(1)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                log_prob = model.log_prob(y, labels) + log_det_pre
                nll = -log_prob.mean()
            scaler.scale(nll).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            running_loss += nll.item()

        train_nll = running_loss / max(1, len(train_loader))
        train_bpd = train_nll / (28 * 28 * math.log(2.0))

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val_nll, val_bpd = evaluate(model, val_loader, device)

        epoch_time = time.time() - t0
        print(
            f"epoch {epoch:02d} | train_nll {train_nll:.4f} | train_bpd {train_bpd:.4f} "
            f"| val_nll {val_nll:.4f} | val_bpd {val_bpd:.4f} | epoch_time {epoch_time:.2f}s"
        )

        save_checkpoint(
            path=os.path.join(cfg.out_dir, "last.pt"),
            model=model,
            optimizer=opt,
            epoch=epoch,
            cfg=cfg,
            best_val_bpd=min(best_val_bpd, val_bpd),
            ema=ema,
        )

        if cfg.save_best and val_bpd < best_val_bpd:
            best_val_bpd = val_bpd
            save_checkpoint(
                path=os.path.join(cfg.out_dir, "best.pt"),
                model=model,
                optimizer=opt,
                epoch=epoch,
                cfg=cfg,
                best_val_bpd=best_val_bpd,
                ema=ema,
            )

        if epoch % cfg.sample_every == 0:
            with torch.no_grad():
                labels = torch.arange(10, device=device).repeat_interleave(8)
                x_sample = model.sample(labels=labels, temperature=0.85)
                save_image(
                    x_sample,
                    os.path.join(cfg.out_dir, f"sample_epoch_{epoch:02d}.png"),
                    nrow=10,
                )

        model.load_state_dict(backup, strict=True)

    print("training complete")


if __name__ == "__main__":
    main()
