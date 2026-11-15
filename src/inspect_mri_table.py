import argparse
import redivis


ORGANIZATION = "AIMI"
DATASET_REF = "mrnet_knee_mri_s:4a2c"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a Redivis table to determine whether it indexes MRI files."
    )
    parser.add_argument(
        "--table",
        required=True,
        help="Table reference from the output of python -m src.list_tables",
    )
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()

    dataset = redivis.organization(ORGANIZATION).dataset(DATASET_REF)
    table = dataset.table(args.table)

    print("Table reference:", args.table)

    # Fetch and print metadata.
    table.get()
    print("\nTABLE METADATA:")
    print(table.properties)

    print("\nVARIABLES:")
    variables = table.list_variables()
    for variable in variables:
        print(variable.properties)

    print("\nTRYING TO LIST A FEW FILES:")
    try:
        files = table.list_files(max_results=args.max_files)
        print("Files returned:", len(files))
        for file in files:
            print("Name:", file.name)
            print("Path:", file.path)
            print("Metadata:", file.properties)
    except Exception as exc:
        print(
            "This table could not be listed as a file-index table. "
            "It may be a regular tabular table, or may require a file-ID variable."
        )
        print("Redivis response:", repr(exc))


if __name__ == "__main__":
    main()
