import pandas as pd

import redivis


def main():
    # Connect to the MRNet dataset
    organization = redivis.organization("AIMI")
    dataset = organization.dataset("mrnet_knee_mri_s")

    # Label tables for both training and validation
    splits = {
        "train": {
            "abnormality": "train_abnormal",
            "ACL tear": "train_acl",
            "meniscus tear": "train_meniscus",
        },
        "valid": {
            "abnormality": "valid_abnormal",
            "ACL tear": "valid_acl",
            "meniscus tear": "valid_meniscus",
        },
    }

    # Inspect each table
    for split_name, tasks in splits.items():
        print(f"\n========== {split_name.upper()} ==========")

        for task, table_name in tasks.items():
            print(f"\n--- {task} ---")

            table = dataset.table(table_name)
            df = table.to_pandas_dataframe()

            print("Shape:", df.shape)
            print("Columns:", df.columns.tolist())
            print("Missing values:", df.isnull().sum().sum())
            print("Label counts:")
            print(df.iloc[:, 1].value_counts().sort_index())


if __name__ == "__main__":
    main()
