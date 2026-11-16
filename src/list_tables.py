import redivis

ORGANIZATION = "AIMI"
DATASET_REF = "mrnet_knee_mri_s:4a2c"


def main():
    dataset = redivis.organization(ORGANIZATION).dataset(DATASET_REF)
    print("Tables available in the MRNet dataset:\n")
    for table in dataset.list_tables():
        table.get()
        print("Name:", getattr(table, "name", "unknown"))
        print("Reference:", getattr(table, "scoped_reference", "unknown"))
        print("Description:", (table.properties or {}).get("description", ""))
        print("-" * 60)


if __name__ == "__main__":
    main()
