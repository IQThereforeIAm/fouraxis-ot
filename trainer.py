"""
trainer.py – first-pass full training loop
====================================================
This is **still a scaffold** but it already runs a *real* training loop
using PyTorch so you can plug‑in your own model, dataset and loss
without changing the plumbing.

To replace the stubs:
* Implement `build_dataloader()` for your dataset.
* Replace `DummyModel` with your model class.
* Adjust the optimiser / scheduler as needed.

Run from the repo root:
    python -m trainer --data ./data --epochs 20 --batch-size 128
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

###############################################################################
# ↓↓↓                     DATA + MODEL PLACEHOLDERS                       ↓↓↓
###############################################################################

class DummyDataset(Dataset):
    """Minimal dataset that yields random (x, y) pairs.

    Replace with your real Dataset subclass.
    """

    def __init__(self, n_samples: int = 10_000, n_features: int = 32) -> None:
        self.x = torch.randn(n_samples, n_features)
        self.y = torch.randint(0, 2, (n_samples,))

    def __len__(self) -> int:  # type: ignore[override]
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.x[idx], self.y[idx]


def build_dataloader(batch_size: int, num_workers: int = 0) -> DataLoader:
    """Build and return the training DataLoader.

    TODO: Replace DummyDataset with your real dataset loader.
    """
    ds = DummyDataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class DummyModel(nn.Module):
    """Simple MLP – swap this with your model implementation."""

    def __init__(self, n_features: int = 32, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

###############################################################################
# ↑↑↑                           PLACEHOLDERS END                           ↑↑↑
###############################################################################


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimiser: optim.Optimizer, device: torch.device, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimiser.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * inputs.size(0)
        if step % 100 == 0:
            logging.info(f"Epoch {epoch} | step {step}/{len(loader)} | loss {loss.item():.4f}")

    return running_loss / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model on dataset")
    parser.add_argument("--data", type=Path, default=Path("./data"), help="Path to data root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Build components
    loader = build_dataloader(args.batch_size)
    sample_features = next(iter(loader))[0].shape[1]
    model = DummyModel(n_features=sample_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, criterion, optimiser, device, epoch)
        logging.info(f"Epoch {epoch} finished | avg loss {avg_loss:.4f}")

    # Save final checkpoint
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "model_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    logging.info(f"Saved checkpoint to {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
