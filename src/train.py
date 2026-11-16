import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import MRNetDataset, TASKS
from src.model import MRNetClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, task_index):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"][:, task_index].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, task_index):
    model.eval()
    losses, targets, probabilities = 0.0, [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"][:, task_index].to(device)
        logits = model(images)
        losses += criterion(logits, labels).item() * labels.size(0)
        targets.extend(labels.cpu().numpy().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    metrics = {
        "loss": losses / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(targets, probabilities)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the three MRNet knee MRI classifiers.")
    parser.add_argument("--data-root", default="data/MRNet-v1.0")
    parser.add_argument("--train-labels", default="data/labels/train_labels_all.csv")
    parser.add_argument("--valid-labels", default="data/labels/valid_labels_all.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-slices", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on {device}")

    train_set = MRNetDataset(args.data_root, "train", args.train_labels, args.image_size, args.num_slices, True)
    valid_set = MRNetDataset(args.data_root, "valid", args.valid_labels, args.image_size, args.num_slices, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    print(f"Training examinations: {len(train_set)} | Validation examinations: {len(valid_set)}")

    for task_index, task in enumerate(TASKS):
        print(f"\n{'=' * 16} {task.upper()} {'=' * 16}")
        model = MRNetClassifier(pretrained=args.pretrained).to(device)
        labels = train_set.labels[task].to_numpy()
        positives = max(float(labels.sum()), 1.0)
        negatives = max(float(len(labels) - labels.sum()), 1.0)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_auc = -1.0
        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task_index)
            metrics = validate(model, valid_loader, criterion, device, task_index)
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            auc = metrics["roc_auc"] if metrics["roc_auc"] is not None else -1.0
            print(
                f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
                f"| val_loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f} "
                f"| f1={metrics['f1']:.4f} | auc={metrics['roc_auc']}"
            )
            if auc > best_auc:
                best_auc = auc
                torch.save(
                    {"task": task, "model_state_dict": model.state_dict(), "epoch": epoch,
                     "validation_metrics": metrics, "args": vars(args)},
                    output_dir / f"{task}_best.pth",
                )

        with open(output_dir / f"{task}_history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    print(f"\nTraining finished. Best checkpoints and histories are in {output_dir}.")


if __name__ == "__main__":
    main()
