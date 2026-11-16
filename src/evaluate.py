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

from src.dataset import MRNetDataset, TASKS
from src.model import MRNetClassifier


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate saved MRNet task classifiers.")
    parser.add_argument("--data-root", default="data/MRNet-v1.0")
    parser.add_argument("--labels", default="data/labels/valid_labels_all.csv")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-slices", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MRNetDataset(args.data_root, "valid", args.labels, args.image_size, args.num_slices, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for task_index, task in enumerate(TASKS):
        checkpoint_path = Path(args.checkpoint_dir) / f"{task}_best.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the models first.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = MRNetClassifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        targets, probabilities = [], []

        for batch in loader:
            logits = model(batch["image"].to(device))
            targets.extend(batch["labels"][:, task_index].numpy().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        predictions = (np.asarray(probabilities) >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(targets, predictions)),
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "recall": float(recall_score(targets, predictions, zero_division=0)),
            "f1": float(f1_score(targets, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
            "classification_report": classification_report(targets, predictions, zero_division=0, output_dict=True),
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
