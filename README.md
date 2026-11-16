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

We have also added the first version of the image loader, ResNet18 model, training script and evaluation script. The model code passed a small test using sample data. The MRI files and label tables have been downloaded. We have added scripts to summarize the dataset, train the three classifiers, evaluate them and make predictions for an individual examination. Full model training and evaluation are being left until the rest of the project code is ready.

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
  - `prepare_data.py`
  - `predict.py`
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

## Preparing the data

Before training, the data summary script checks which labelled examinations have all three MRI planes, reports the class counts and saves the results in `reports/data_summary.json`. Run `python -m src.prepare_data`.

## Training

The training script builds a separate classifier for abnormality, ACL tear and meniscus tear. It uses class-weighted loss, a learning-rate scheduler and early stopping. On a CUDA GPU it also enables mixed-precision training. The best checkpoints and epoch histories are saved in `models/`.

When we are ready to train, run `python -m src.train --data-root data/MRNet-v1.0 --epochs 10 --batch-size 2 --pretrained`. The batch size and number of slices can be reduced if GPU memory is limited.

## Evaluation

After training, run `python -m src.evaluate --data-root data/MRNet-v1.0`.

The metrics, ROC curves and confusion matrices are saved in the `reports/` folder.

## Testing the code

We added a small test that creates temporary sample MRI arrays and checks whether the data loader and model work. Run `python -m src.smoke_test`.

This test passed with sample data. It only checks that the code runs; it does not mean the models have been trained on the actual MRI scans.

## Making a prediction

After the three model checkpoints have been trained, a prediction for one examination can be generated with `python -m src.predict --exam-id 0000 --split valid`. The script prints the predicted label and probability for each task.

## Current status

The label extraction and MRI downloads are complete. The data preparation, model, training, evaluation and single-examination prediction code is in place. The remaining major work is to train the models, review their validation results and improve them where needed.

