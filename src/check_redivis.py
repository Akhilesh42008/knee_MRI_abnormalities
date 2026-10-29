import redivis

# Connect to the dataset organization
organization = redivis.organization("AIMI")

# Select the MRNet knee MRI dataset
dataset = organization.dataset("mrnet_knee_mri_s")

# List the available tables
tables = dataset.list_tables()

print("MRNet dataset connection successful!")
print("\nAvailable tables:")

for table in tables:
    print(table)