import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import TASKS


def main():
    parser = argparse.ArgumentParser(description="Plot the training history for all three tasks.")
    parser.add_argument("--history-dir", default="models")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        path = history_dir / f"{task}_history.json"
        if not path.is_file():
            raise FileNotFoundError(f"Training history not found: {path}")

        history = json.loads(path.read_text(encoding="utf-8"))
        epochs = [item["epoch"] for item in history]

        fig, ax = plt.subplots()
        ax.plot(epochs, [item["train_loss"] for item in history], marker="o", label="Training loss")
        ax.plot(epochs, [item["loss"] for item in history], marker="o", label="Validation loss")
        ax.set_title(f"{task.upper()} loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_loss_history.png", dpi=150)
        plt.close(fig)

        auc_values = [item.get("roc_auc") for item in history]
        if any(value is not None for value in auc_values):
            fig, ax = plt.subplots()
            valid_epochs = [epoch for epoch, value in zip(epochs, auc_values) if value is not None]
            valid_auc = [value for value in auc_values if value is not None]
            ax.plot(valid_epochs, valid_auc, marker="o", label="Validation ROC-AUC")
            ax.set_title(f"{task.upper()} validation ROC-AUC")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("ROC-AUC")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / f"{task}_auc_history.png", dpi=150)
            plt.close(fig)

    print(f"Training plots saved to {output_dir}")


if __name__ == "__main__":
    main()
