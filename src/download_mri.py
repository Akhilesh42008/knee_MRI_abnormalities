import argparse
from pathlib import Path

import redivis

ORGANIZATION = "AIMI"
DATASET_REF = "mrnet_knee_mri_s:4a2c"


def main():
    parser = argparse.ArgumentParser(
        description="Download MRI volumes from the Redivis file-index table."
    )
    parser.add_argument("--table", required=True, help="MRI file-index table reference")
    parser.add_argument("--output", default="data/MRNet-v1.0")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for a small first download; omit to download every file.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = redivis.organization(ORGANIZATION).dataset(DATASET_REF)
    table = dataset.table(args.table)
    table.get()

    print(f"Downloading from: {args.table}")
    print(f"Destination: {output.resolve()}")
    if args.max_files:
        print(f"File limit: {args.max_files}")
    else:
        print("Download limit: none")

    directory = table.to_directory(max_results=args.max_files)
    directory.download(str(output), max_results=args.max_files, progress=True)

    print("\nDownload finished.")
    print(f"Files are in: {output.resolve()}")
    print("The files should retain their folders and examination filenames.")


if __name__ == "__main__":
    main()
