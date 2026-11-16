import argparse
import json

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
    parser.add_argument("--num-slices", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    labels_path = args.labels or (
        "data/labels/train_labels_all.csv" if args.split == "train"
        else "data/labels/valid_labels_all.csv"
    )
    dataset = MRNetDataset(
        args.data_root, args.split, labels_path,
        image_size=args.image_size, num_slices=args.num_slices, training=False,
    )
    wanted_id = str(args.exam_id).zfill(4) if str(args.exam_id).isdigit() else str(args.exam_id)
    sample_index = next(
        (index for index, sample in enumerate(dataset.samples) if sample[0].zfill(4) == wanted_id),
        None,
    )
    if sample_index is None:
        raise ValueError(f"Examination {args.exam_id} was not found in the {args.split} dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = dataset[sample_index]
    results = {}

    for task_index, task in enumerate(TASKS):
        checkpoint_path = f"{args.checkpoint_dir}/{task}_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
