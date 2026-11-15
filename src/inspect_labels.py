from pathlib import Path
import numbers

import pandas as pd
import redivis


ORGANIZATION = "AIMI"
DATASET_REF = "mrnet_knee_mri_s:4a2c"

SPLITS = {
    "train": {
        "abnormal": "train_abnormal:4v5w",
        "acl": "train_acl:v9mj",
        "meniscus": "train_meniscus:3x4c",
    },
    "valid": {
        "abnormal": "valid_abnormal:851r",
        "acl": "valid_acl:sqwh",
        "meniscus": "valid_meniscus:nqng",
    },
}

LABEL_COLUMNS = ["abnormal", "acl", "meniscus"]
OUTPUT_DIR = Path("data") / "labels"


def normalize_exam_id(value):
    """Normalize numeric IDs without turning 12.0 into the string '12.0'."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    return str(value).strip()


def get_clean_label_table(dataset, table_reference, task_name, split_name):
    table = dataset.table(table_reference)
    df = table.to_pandas_dataframe()

    if df.shape[1] < 2:
        raise ValueError(
            f"{split_name}/{task_name}: expected at least 2 columns, "
            f"but found {df.columns.tolist()}"
        )

    print(f"\n--- {split_name.upper()} / {task_name} ---")
    print("Original columns:", df.columns.tolist())
    print("Original shape:", df.shape)

    # This preserves the assumption in your existing script:
    # first column = exam ID, second column = label.
    # Verify the printed column names are correct for all six Redivis tables.
    df = df.iloc[:, :2].copy()
    df.columns = ["exam_id", "label"]

    df["exam_id"] = df["exam_id"].map(normalize_exam_id)

    missing_ids = int(df["exam_id"].isna().sum())
    missing_labels = int(df["label"].isna().sum())
    print("Missing exam IDs:", missing_ids)
    print("Missing labels:", missing_labels)

    df = df.dropna(subset=["exam_id", "label"]).copy()

    duplicate_count = int(df["exam_id"].duplicated(keep=False).sum())
    print("Rows involved in duplicate exam IDs:", duplicate_count)

    # Duplicates with conflicting labels must be investigated, not silently removed.
    conflicts = df.groupby("exam_id")["label"].nunique(dropna=True)
    conflicting_ids = conflicts[conflicts > 1].index.tolist()
    if conflicting_ids:
        raise ValueError(
            f"{split_name}/{task_name}: conflicting labels for exam IDs "
            f"{conflicting_ids[:10]}"
        )

    df = df.drop_duplicates(subset=["exam_id"], keep="first").reset_index(drop=True)

    print("Clean shape:", df.shape)
    print("Unique exam IDs:", df["exam_id"].nunique())
    print("Label counts:")
    print(df["label"].value_counts(dropna=False).sort_index())
    print("First five rows:")
    print(df.head())

    # Report unexpected label values without assuming the provider's encoding.
    values = sorted(df["label"].astype(str).unique().tolist())
    print("Distinct label values:", values)

    raw_output = OUTPUT_DIR / f"{split_name}_{task_name}.csv"
    df.to_csv(raw_output, index=False)
    print("Saved:", raw_output)

    return df.rename(columns={"label": task_name})


def build_split(dataset, split_name, table_map):
    task_frames = {}
    for task_name, table_reference in table_map.items():
        task_frames[task_name] = get_clean_label_table(
            dataset, table_reference, task_name, split_name
        )

    # Outer joins preserve every exam ID and expose missing task labels.
    merged = task_frames["abnormal"].merge(
        task_frames["acl"], on="exam_id", how="outer", validate="one_to_one"
    ).merge(
        task_frames["meniscus"], on="exam_id", how="outer", validate="one_to_one"
    )

    merged = merged.sort_values("exam_id").reset_index(drop=True)
    merged_output = OUTPUT_DIR / f"{split_name}_labels_all.csv"
    merged.to_csv(merged_output, index=False)

    missing_by_task = merged[LABEL_COLUMNS].isna().sum()
    complete = merged.dropna(subset=LABEL_COLUMNS).copy()
    complete_output = OUTPUT_DIR / f"{split_name}_labels.csv"
    complete.to_csv(complete_output, index=False)

    print(f"\n========== {split_name.upper()} MERGE SUMMARY ==========")
    print("All unique exam IDs across tasks:", len(merged))
    print("Missing labels by task:")
    print(missing_by_task)
    print("Exams with all three labels:", len(complete))
    print("Saved all IDs (including incomplete labels):", merged_output)
    print("Saved complete three-label CSV:", complete_output)

    for col in LABEL_COLUMNS:
        unique_values = set(complete[col].dropna().astype(str).unique())
        if not unique_values.issubset({"0", "1", "0.0", "1.0"}):
            print(
                f"WARNING: {col} does not look strictly binary. "
                f"Observed values: {sorted(unique_values)}"
            )

    return merged


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    organization = redivis.organization(ORGANIZATION)
    dataset = organization.dataset(DATASET_REF)

    for split_name, table_map in SPLITS.items():
        build_split(dataset, split_name, table_map)

    print("\nLabel extraction completed.")
    print("Next: inspect the MRI file/index table before implementing image loading.")


if __name__ == "__main__":
    main()
