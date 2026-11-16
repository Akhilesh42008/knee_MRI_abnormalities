import argparse
import json
from pathlib import Path

import torch

from src.dataset import MRNetDataset, TASKS
from src.model import MRNetClassifier


def main():
    parser = argparse.ArgumentParser(description="Predict the three MRNet labels for one examination.")
    parser.add_argument("--exam-id", required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--data-root", default="data/MRNet-v1.0")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--checkpoint-dir", default="models")
    args = parser.parse_args()

    labels_path = args.labels or (
        "data/labels/train_labels_all.csv" if args.split == "train"
        else "data/labels/valid_labels_all.csv"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = {}
    for task in TASKS:
        path = checkpoint_dir / f"{task}_best.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}. Train the models first.")
        checkpoints[task] = torch.load(path, map_location=device, weights_only=False)

    model_args = checkpoints[TASKS[0]].get("args", {})
    dataset = MRNetDataset(
        args.data_root, args.split, labels_path,
        image_size=model_args.get("image_size", 224),
        num_slices=model_args.get("num_slices", 16),
        training=False,
    )
    wanted_id = str(args.exam_id).zfill(4) if str(args.exam_id).isdigit() else str(args.exam_id)
    sample_index = next(
        (index for index, sample in enumerate(dataset.samples)
         if (sample[0].zfill(4) if sample[0].isdigit() else sample[0]) == wanted_id),
        None,
    )
    if sample_index is None:
        raise ValueError(f"Examination {args.exam_id} was not found in the {args.split} dataset.")

    sample = dataset[sample_index]
    results = {}
    for task in TASKS:
        checkpoint = checkpoints[task]
        model = MRNetClassifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logit = model(sample["image"].unsqueeze(0).to(device))
            probability = torch.sigmoid(logit).item()
        results[task] = {
            "probability": round(probability, 4),
            "predicted_label": int(probability >= 0.5),
        }

    print(json.dumps({
        "exam_id": sample["exam_id"],
        "predictions": results,
        "note": "Experimental model output; not for clinical diagnosis.",
    }, indent=2))


if __name__ == "__main__":
    main()
