import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import PLANES, TASKS


def normalise_id(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(4) if value.isdigit() else value


def inspect_split(data_root, split, labels_path):
    labels = pd.read_csv(labels_path)
    required = {"exam_id", *TASKS}
    missing_columns = required.difference(labels.columns)
    if missing_columns:
        raise ValueError(f"{labels_path} is missing columns: {sorted(missing_columns)}")

    labels["exam_id"] = labels["exam_id"].map(normalise_id)
    report = {
        "split": split,
        "label_rows": int(len(labels)),
        "label_duplicates": int(labels["exam_id"].duplicated().sum()),
        "tasks": {},
        "planes": {},
        "complete_examinations": 0,
        "missing_examinations": [],
        "invalid_volumes": [],
    }

    for task in TASKS:
        values = pd.to_numeric(labels[task], errors="coerce")
        report["tasks"][task] = {
            "negative": int((values == 0).sum()),
            "positive": int((values == 1).sum()),
            "missing_or_invalid": int(values.isna().sum()),
        }

    plane_ids = {}
    for plane in PLANES:
        folder = Path(data_root) / split / plane
        paths = list(folder.glob("*.npy")) if folder.exists() else []
        ids = {normalise_id(path.stem) for path in paths}
        plane_ids[plane] = ids
        report["planes"][plane] = {
            "files_found": len(paths),
            "folder_exists": folder.exists(),
        }

    labelled_ids = set(labels["exam_id"])
    all_ids = set.intersection(*(plane_ids[plane] for plane in PLANES))
    report["complete_examinations"] = len(labelled_ids & all_ids)
    report["missing_examinations"] = sorted(labelled_ids - all_ids)[:50]

    for plane in PLANES:
        report["planes"][plane]["without_label"] = len(plane_ids[plane] - labelled_ids)
        report["planes"][plane]["missing_from_plane"] = len(labelled_ids - plane_ids[plane])

    return report


def main():
    parser = argparse.ArgumentParser(description="Prepare and summarize MRNet data before training.")
    parser.add_argument("--data-root", default="data/MRNet-v1.0")
    parser.add_argument("--train-labels", default="data/labels/train_labels_all.csv")
    parser.add_argument("--valid-labels", default="data/labels/valid_labels_all.csv")
    parser.add_argument("--output", default="reports/data_summary.json")
    args = parser.parse_args()

    reports = [
        inspect_split(args.data_root, "train", args.train_labels),
        inspect_split(args.data_root, "valid", args.valid_labels),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    for report in reports:
        print(f"\n{report['split'].upper()}")
        print(f"Label rows: {report['label_rows']}")
        print(f"Complete examinations with all three planes: {report['complete_examinations']}")
        for task, counts in report["tasks"].items():
            print(f"{task}: negative={counts['negative']}, positive={counts['positive']}")
        if report["missing_examinations"]:
            print("Some labelled exams are missing one or more MRI planes; see the JSON report.")

    print(f"\nSummary saved to {output}")


if __name__ == "__main__":
    main()
