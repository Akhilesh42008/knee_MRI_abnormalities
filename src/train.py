import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import MRNetDataset, TASKS
from src.model import MRNetClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, task_index, scaler, epoch, epochs, task):
    started = time.perf_counter()
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"{task} train {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"][:, task_index].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset), time.perf_counter() - started


@torch.no_grad()
def validate(model, loader, criterion, device, task_index, task, epoch, epochs):
    started = time.perf_counter()
    model.eval()
    losses, targets, probabilities = 0.0, [], []
    progress = tqdm(loader, desc=f"{task} valid {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"][:, task_index].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        losses += loss.item() * labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
        targets.extend(labels.cpu().numpy().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    metrics = {
        "loss": losses / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics, time.perf_counter() - started


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
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_slices < 1:
        parser.error("epochs, batch-size and num-slices must be positive integers")

    overall_started = time.perf_counter()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on {device}")

    train_set = MRNetDataset(args.data_root, "train", args.train_labels, args.image_size, args.num_slices, True)
    valid_set = MRNetDataset(args.data_root, "valid", args.valid_labels, args.image_size, args.num_slices, False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    print(f"Training examinations: {len(train_set)} | Validation examinations: {len(valid_set)}")

    for task_index, task in enumerate(TASKS):
        task_started = time.perf_counter()
        print(f"\n{'=' * 16} {task.upper()} {'=' * 16}")
        model = MRNetClassifier(pretrained=args.pretrained).to(device)
        labels = train_set.labels[task].to_numpy()
        positives = max(float(labels.sum()), 1.0)
        negatives = max(float(len(labels) - labels.sum()), 1.0)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_auc = -1.0
        best_loss = float("inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, train_seconds = train_epoch(model, train_loader, optimizer, criterion, device, task_index, scaler, epoch, args.epochs, task)
            metrics, valid_seconds = validate(model, valid_loader, criterion, device, task_index, task, epoch, args.epochs)
            epoch_seconds = time.perf_counter() - epoch_started
            scheduler.step(metrics["loss"])
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_seconds": train_seconds,
                "validation_seconds": valid_seconds,
                "epoch_seconds": epoch_seconds,
                **metrics,
            })
            auc = metrics["roc_auc"] if metrics["roc_auc"] is not None else -1.0
            print(
                f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
                f"| val_loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f} "
                f"| f1={metrics['f1']:.4f} | auc={metrics['roc_auc']} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            improved = auc > best_auc if metrics["roc_auc"] is not None else metrics["loss"] < best_loss
            if improved:
                best_auc = max(best_auc, auc)
                best_loss = min(best_loss, metrics["loss"])
                epochs_without_improvement = 0
                torch.save(
                    {
                        "task": task,
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "validation_metrics": metrics,
                        "args": vars(args),
                    },
                    output_dir / f"{task}_best.pth",
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping {task}: validation did not improve for {args.patience} epochs.")
                break

        task_seconds = time.perf_counter() - task_started
        with open(output_dir / f"{task}_history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        print(f"{task.upper()} total time: {task_seconds / 60:.1f} minutes ({task_seconds / 3600:.2f} hours)")

    overall_seconds = time.perf_counter() - overall_started
    print(f"\nTotal training time: {overall_seconds / 60:.1f} minutes ({overall_seconds / 3600:.2f} hours)")
    print(f"Best checkpoints and histories are in {output_dir}.")


if __name__ == "__main__":
    main()
