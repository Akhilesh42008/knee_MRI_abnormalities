import argparse
from pathlib import Path

import redivis

ORGANIZATION = "AIMI"
DATASET_REF = "mrnet_knee_mri_s:4a2c"

SPLITS = {
    "train": "train:5wjf",
    "valid": "valid:zcmk",
}


def download_split(dataset, split, table_ref, output_root, max_files=None):
    table = dataset.table(table_ref)
    table.get()

    destination = output_root / split
    destination.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {split} MRI files")
    print(f"Redivis table: {table_ref}")
    print(f"Destination: {destination.resolve()}")

    files = table.list_files(max_results=max_files)
    expected_count = len(files)
    print(f"Files to download: {expected_count}")

    directory = table.to_directory()
    directory.download(
        str(destination),
        max_results=max_files,
        progress=True,
        overwrite=True,
        max_parallelization=2,
        max_concurrency=4,
    )

    downloaded_count = sum(1 for path in destination.rglob("*") if path.is_file())
    print(f"{split}: {downloaded_count}/{expected_count} files present")

    if downloaded_count < expected_count:
        raise RuntimeError(
            f"{split} download is incomplete: expected {expected_count} files, "
            f"but found {downloaded_count}. Run the command again; existing files "
            "will be overwritten and the count will be checked again."
        )

    print(f"{split} download finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Download the MRNet training and validation MRI files from Redivis."
    )
    parser.add_argument("--output", default="data/MRNet-v1.0")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional per-split file limit. Leave unset to download all files.",
    )
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = redivis.organization(ORGANIZATION).dataset(DATASET_REF)

    for split, table_ref in SPLITS.items():
        download_split(dataset, split, table_ref, output_root, args.max_files)

    print("\nBoth MRI splits have been downloaded completely.")
    print(f"Dataset location: {output_root.resolve()}")
    print("Keep the downloaded plane folders and examination filenames unchanged.")


if __name__ == "__main__":
    main()
