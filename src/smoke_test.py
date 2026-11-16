from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from src.dataset import MRNetDataset
from src.model import MRNetClassifier


def main():
    print("Checking the dataset loader and model with a tiny synthetic example.")
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for split in ("train", "valid"):
            for plane in ("axial", "coronal", "sagittal"):
                folder = root / split / plane
                folder.mkdir(parents=True)
                np.save(folder / "0001.npy", np.random.rand(8, 64, 64).astype("float32"))

        labels = pd.DataFrame({
            "exam_id": ["0001"],
            "abnormal": [1],
            "acl": [0],
            "meniscus": [1],
        })
        train_csv = root / "train.csv"
        valid_csv = root / "valid.csv"
        labels.to_csv(train_csv, index=False)
        labels.to_csv(valid_csv, index=False)

        dataset = MRNetDataset(root, "train", train_csv, image_size=64, num_slices=4)
        sample = dataset[0]
        assert sample["image"].shape == (3, 4, 3, 64, 64)
        assert sample["labels"].shape == (3,)

        model = MRNetClassifier(pretrained=False)
        model.eval()
        with torch.no_grad():
            prediction = model(sample["image"].unsqueeze(0))
        assert prediction.shape == (1,)

    print("All checks passed. The loader and model can process a sample examination.")


if __name__ == "__main__":
    main()
