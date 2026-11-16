from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


PLANES = ("axial", "coronal", "sagittal")
TASKS = ("abnormal", "acl", "meniscus")


class MRNetDataset(Dataset):
    """Loads one examination from the three MRNet imaging planes."""

    def __init__(self, data_root, split, labels_csv, image_size=224, num_slices=16, training=False):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.num_slices = num_slices
        self.training = training
        self.labels = pd.read_csv(labels_csv)

        required = {"exam_id", *TASKS}
        missing = required.difference(self.labels.columns)
        if missing:
            raise ValueError(f"Missing columns in {labels_csv}: {sorted(missing)}")

        self.labels["exam_id"] = self.labels["exam_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        self.labels = self.labels.dropna(subset=list(TASKS)).reset_index(drop=True)
        for task in TASKS:
            values = set(pd.to_numeric(self.labels[task], errors="coerce").dropna().unique())
            if not values.issubset({0, 1}):
                raise ValueError(f"{task} must contain binary 0/1 labels; found {values}")

        self.labels[list(TASKS)] = self.labels[list(TASKS)].astype("float32")
        self.samples = []
        for _, row in self.labels.iterrows():
            exam_id = str(row["exam_id"])
            # MRNet filenames commonly use four-digit IDs (for example 0007.npy),
            # while a CSV reader may represent the same ID as "7".
            file_ids = [exam_id]
            if exam_id.isdigit():
                file_ids.append(exam_id.zfill(4))
            paths = {}
            for plane in PLANES:
                folder = self.data_root / split / plane
                matches = [folder / f"{file_id}.npy" for file_id in file_ids]
                paths[plane] = next((path for path in matches if path.is_file()), matches[0])
            if all(path.is_file() for path in paths.values()):
                self.samples.append((exam_id, paths, row[list(TASKS)].to_numpy(dtype=np.float32)))

        if not self.samples:
            example = self.data_root / split / "axial" / "0000.npy"
            raise FileNotFoundError(
                f"No complete examinations found. Expected files such as {example}. "
                "Check --data-root, split names, exam IDs, and the three plane folders."
            )

    def __len__(self):
        return len(self.samples)

    def _select_slices(self, volume):
        if volume.ndim != 3:
            raise ValueError(f"Expected MRI array with shape [slices, height, width], got {volume.shape}")
        count = volume.shape[0]
        if count == 0:
            raise ValueError("An MRI volume contains no slices")
        if count >= self.num_slices:
            if self.training:
                positions = np.linspace(0, count - 1, self.num_slices)
                step = max(1, count // self.num_slices)
                indices = []
                for position in positions:
                    low = max(0, int(position) - step // 2)
                    high = min(count, int(position) + step // 2 + 1)
                    indices.append(random.randrange(low, high) if high > low else int(position))
                indices = np.array(sorted(indices))
            else:
                indices = np.linspace(0, count - 1, self.num_slices).round().astype(int)
            volume = volume[indices]
        else:
            indices = np.linspace(0, count - 1, self.num_slices).round().astype(int)
            volume = volume[indices]

        volume = np.nan_to_num(volume.astype(np.float32), copy=False)
        slices = []
        for image in volume:
            image = torch.from_numpy(image).unsqueeze(0)
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            low, high = image.amin(), image.amax()
            image = (image - low) / (high - low).clamp_min(1e-6)
            image = image.repeat(3, 1, 1)
            slices.append(image)
        return torch.stack(slices)

    def __getitem__(self, index):
        exam_id, paths, labels = self.samples[index]
        views = [self._select_slices(np.load(paths[plane], mmap_mode="r")) for plane in PLANES]
        return {
            "image": torch.stack(views),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "exam_id": exam_id,
        }
