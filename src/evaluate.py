import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, RocCurveDisplay,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import MRNetDataset, TASKS
from src.model import MRNetClassifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate the three saved MRNet classifiers.")
    parser.add_argument("--data-root", default="data/MRNet-v1.0")
    parser.add_argument("--labels", default="data/labels/valid_labels_all.csv")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = {}
    for task in TASKS:
        path = checkpoint_dir / f"{task}_best.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        checkpoints[task] = torch.load(path, map_location=device, weights_only=False)

    model_args = checkpoints[TASKS[0]].get("args", {})
    image_size = model_args.get("image_size", 224)
    num_slices = model_args.get("num_slices", 16)
    dataset = MRNetDataset(
        args.data_root, "valid", args.labels,
        image_size=image_size, num_slices=num_slices, training=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for task_index, task in enumerate(TASKS):
        checkpoint = checkpoints[task]
        task_args = checkpoint.get("args", {})
        if task_args.get("image_size", image_size) != image_size or task_args.get("num_slices", num_slices) != num_slices:
            raise ValueError("The three checkpoints use different input sizes. Evaluate them separately.")
        model = MRNetClassifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        targets, probabilities = [], []

        with torch.inference_mode():
            progress = tqdm(loader, desc=f"{task} evaluation", unit="batch", dynamic_ncols=True)
            for batch in progress:
                logits = model(batch["image"].to(device, non_blocking=True))
                targets.extend(batch["labels"][:, task_index].cpu().numpy().tolist())
                probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        predictions = (np.asarray(probabilities) >= 0.5).astype(int)
        metrics = {
            "checkpoint_epoch": checkpoint.get("epoch"),
            "exam_count": len(targets),
            "accuracy": float(accuracy_score(targets, predictions)),
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "recall": float(recall_score(targets, predictions, zero_division=0)),
            "f1": float(f1_score(targets, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                targets, predictions, labels=[0, 1], target_names=["negative", "positive"],
                zero_division=0, output_dict=True,
            ),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
            RocCurveDisplay.from_predictions(targets, probabilities)
            plt.title(f"{task.upper()} ROC curve")
            plt.tight_layout()
            plt.savefig(output_dir / f"{task}_roc.png", dpi=150)
            plt.close()
        except ValueError:
            metrics["roc_auc"] = None

        summary[task] = metrics
        matrix = np.asarray(metrics["confusion_matrix"])
        fig, ax = plt.subplots()
        ax.imshow(matrix)
        ax.set_title(f"{task.upper()} confusion matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1], ["Negative", "Positive"])
        ax.set_yticks([0, 1], ["Negative", "Positive"])
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        print(f"\n{task.upper()}: {json.dumps(metrics, indent=2)}")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nReports saved to {output_dir}")


if __name__ == "__main__":
    main()
