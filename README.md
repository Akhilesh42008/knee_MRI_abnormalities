# Knee MRI Classification

A small team project where we are learning how to use deep learning to recognise three findings in knee MRI examinations. We are using the Stanford MRNet dataset and PyTorch.

The project has three separate prediction tasks:

- **Abnormality:** whether the examination is labelled abnormal.
- **ACL:** whether an ACL tear is present.
- **Meniscus:** whether a meniscus tear is present.

We are building this as a learning project, so the code is kept in separate files and the results will be reported from actual validation runs. This is an experimental educational model, not a medical diagnostic tool.

## Team

- Akhilesh Bethireddy (SC25B141)
- Gireesh Bhavani Molkalapalli (SC25B145)
- Lokesh Jallu (SC25B150)

## Where the project stands

The label-inspection scripts are in place. We have now added the first end-to-end modelling pipeline: a dataset loader, a shared ResNet18-based model, training for each of the three tasks, and a validation evaluation script.

The next thing we need to verify on the machine that has dataset access is the MRI file export and folder layout. The model code expects one NumPy volume per examination and imaging plane. We have not reported model accuracy yet because the pipeline still needs to be run against the actual MRI files.

## Project layout

```text
.
├── data/
│   └── labels/                 # Generated label CSVs (not committed)
├── models/                     # Saved model checkpoints (not committed)
├── reports/                    # Evaluation plots and metrics (not committed)
├── src/
│   ├── check_redivis.py
│   ├── inspect_labels.py
│   ├── inspect_mri_table.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── main.py
└── requirements.txt
```

## Setup

Use Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, install the build that matches the CUDA support on your machine using the official PyTorch installation selector.

## Dataset preparation

The label inspection script uses the Redivis dataset references already recorded in the source code. Run it in an environment where your Redivis access is configured:

```bash
python -m src.inspect_labels
```

It writes the individual task tables and merged label files into `data/labels/`. The training code reads the merged files and expects these columns:

- `exam_id`
- `abnormal`
- `acl`
- `meniscus`

The MRI volumes are expected to be exported as NumPy arrays with shape `[slices, height, width]`, one file per exam and plane:

```text
data/MRNet-v1.0/
├── train/
│   ├── axial/0000.npy
│   ├── coronal/0000.npy
│   └── sagittal/0000.npy
└── valid/
    ├── axial/0000.npy
    ├── coronal/0000.npy
    └── sagittal/0000.npy
```

Use the actual examination IDs from the label tables as filenames (the `0000.npy` above is only an example). Keep the original dataset files private and follow the dataset's access and usage conditions. The repository intentionally ignores local data and model files.

Before training, make sure both split label files exist:

- `data/labels/train_labels_all.csv`
- `data/labels/valid_labels_all.csv`

The loader skips examinations that do not have all three plane files and all three task labels. It raises an error if it cannot find any complete examinations.

## Quick pipeline check

Before using the real MRI files, run a tiny synthetic check. It creates temporary example volumes, checks that the loader returns the expected tensor shape, and sends one example through the model:

```bash
python -m src.smoke_test
```

This checks the code path and installed dependencies; it does not train a model or measure accuracy.

## Train the models

From the repository root:

```bash
python -m src.train --data-root data/MRNet-v1.0 --epochs 10 --batch-size 2
```

This trains one model per task and saves the best validation checkpoint and a JSON training history in `models/`. To start from ImageNet-pretrained ResNet18 weights, add `--pretrained`. The first run with that option may download weights.

Useful options include `--num-slices`, `--image-size`, `--lr`, and `--workers`. Start with a small batch size if GPU memory is limited.

## Evaluate

After training:

```bash
python -m src.evaluate --data-root data/MRNet-v1.0
```

The script writes per-task metrics, ROC curves (when both classes are present), and confusion matrices into `reports/`.

## A note on the current baseline

The first version samples a fixed number of slices from each MRI volume, encodes the slices with ResNet18, pools the slice features, and combines the three imaging planes for a binary prediction. Each task is trained separately with a class-weighted binary loss.

This is a baseline to get the full workflow working. It is not guaranteed to match the original MRNet paper's preprocessing or performance. We will tune it after checking the exported file format, class balance, and validation behaviour. The validation split is used for model selection and should not be described as an independent test set.

## Current limitations / next checks

- Confirm that the Redivis MRI file table can be exported into the NumPy layout described above.
- Verify the examination IDs and split membership against the actual MRI files.
- Run a small end-to-end smoke test before a full training job.
- Review class balance and metrics before making any performance claims.
- Add an independent test evaluation if an authorised test split is available.

