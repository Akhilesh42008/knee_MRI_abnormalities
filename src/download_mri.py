import argparse
from pathlib import Path

import redivis

ORGANIZATION = "AIMI"
DATASET_REF = "mrnet_knee_mri_s:4a2c"


def main():
    parser = argparse.ArgumentParser(
        description="Download the MRI files from the Redivis file-index table."
    )
    parser.add_argument("--table", required=True, help="File-index table reference from src.list_tables")
    parser.add_argument("--output", default="data/MRNet-v1.0", help="Folder for the downloaded MRI files")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = redivis.organization(ORGANIZATION).dataset(DATASET_REF)
    table = dataset.table(args.table)
    table.get()

    print("Selected table:", args.table)
    print("Downloading MRI files to:", output.resolve())
    print("This can take a while because the MRI volumes are large.")

    directory = table.to_directory()
    directory.download(str(output))

    print("\nDownload finished.")
    print("Files were saved under:", output.resolve())
    print("Keep the original folder structure; the training loader expects split/plane/exam_id.npy.")


if __name__ == "__main__":
    main()
