# Knee MRI Abnormalities

This is our team project for detecting abnormalities in knee MRI scans using deep learning and PyTorch. We are working with the Stanford MRNet dataset.

We are building models for three tasks:

- Abnormality detection
- ACL tear detection
- Meniscus tear detection

## Team members

- Akhilesh Bethireddy (SC25B141)
- Gireesh Bhavani Molkalapalli (SC25B145)
- Lokesh Jallu (SC25B150)

## Project progress

We connected to the MRNet dataset through Redivis and created scripts to collect and inspect the training and validation labels. The labels for all three tasks are combined using the examination ID.

We have also added the first version of the image loader, ResNet18 model, training script and evaluation script. The model code passed a small test using sample data. Training on the real MRI scans is still to be done.

## Folder structure

The project files are arranged like this:

- `data/`
  - `labels/`
  - `MRNet-v1.0/`
- `models/`
- `reports/`
- `src/`
  - `check_redivis.py`
  - `inspect_labels.py`
  - `inspect_mri_table.py`
  - `list_tables.py`
  - `download_mri.py`
  - `dataset.py`
  - `model.py`
  - `train.py`
  - `evaluate.py`
  - `smoke_test.py`
- `main.py`
- `requirements.txt`
- `README.md`

The dataset, model checkpoints and reports stay on the local machine and are not uploaded to GitHub.

## Getting the labels

The label tables are available through Redivis. After setting up Redivis access, run `python -m src.inspect_labels`.

The script saves the individual label files and the combined files in `data/labels/`:

- `train_labels_all.csv`
- `valid_labels_all.csv`

Each combined file contains the examination ID and the labels for abnormality, ACL and meniscus.

## Downloading the MRI files

The MRI file tables are available through Redivis:

- Training table: `train:5wjf`
- Validation table: `valid:zcmk`

Run `python -m src.download_mri` to download both splits.

The files are saved in `data/MRNet-v1.0/`. The download can take some time, so make sure you have enough free disk space.

## Training

Once the MRI files and label CSVs are ready, run `python -m src.train --data-root data/MRNet-v1.0 --epochs 10 --batch-size 2`.

The script trains a separate model for each task. The best checkpoints and training history are saved in the `models/` folder.

To use ImageNet-pretrained ResNet18 weights, add `--pretrained` to the command.

## Evaluation

After training, run `python -m src.evaluate --data-root data/MRNet-v1.0`.

The metrics, ROC curves and confusion matrices are saved in the `reports/` folder.

## Testing the code

We added a small test that creates temporary sample MRI arrays and checks whether the data loader and model work. Run `python -m src.smoke_test`.

This test passed with sample data. It only checks that the code runs; it does not mean the models have been trained on the actual MRI scans.

## Current status

The label extraction and initial model code are ready. We still need to finish downloading the MRI files, train all three models, check their validation results and make improvements if needed.

