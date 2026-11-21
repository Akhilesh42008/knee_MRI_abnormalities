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

We have added the image loader, ResNet18 model, training and evaluation scripts, dataset summary, prediction script and training-history plots. The MRI files and label tables have been downloaded. The data loader and model also passed a small synthetic smoke test.

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
  - `plot_history.py`
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

The training script builds a separate classifier for abnormality, ACL tear and meniscus tear. It uses class-weighted loss, a learning-rate scheduler, early stopping and random slice sampling. On a CUDA GPU it also enables mixed-precision training. The best checkpoints and epoch histories are saved in `models/`.

For a small CPU test run, we used `python -m src.train --epochs 1 --batch-size 1 --num-slices 4 --image-size 128 --workers 0`. This completed one epoch for each of the three tasks and saved checkpoints and training histories in `models/`. It is only an initial run, not a final model. For longer training, increase the epochs gradually and choose settings that the available hardware can handle. On a CUDA GPU, the script can use mixed precision.

## Evaluation

After training, run `python -m src.evaluate`. The evaluator loads the saved checkpoints and runs inference on the validation split. It reports accuracy, precision, recall, F1 and ROC-AUC. The metrics, ROC curves and confusion matrices are saved in the `reports/` folder. Evaluation uses inference mode so gradients are not tracked.

## Testing the code

We added a small test that creates temporary sample MRI arrays and checks whether the data loader and model work. Run `python -m src.smoke_test`.

This test passed with sample data. It only checks that the code runs; it does not mean the models have been trained on the actual MRI scans.

## Training plots

After training, run `python -m src.plot_history` to save loss and ROC-AUC history plots in `reports/`.

## Making a prediction

After the three model checkpoints have been trained, a prediction for one examination can be generated with `python -m src.predict --exam-id 0000 --split valid`. The script prints the predicted label and probability for each task.

## Current status

The label extraction and MRI downloads are complete. The data preparation, model, training, evaluation, prediction and training-plot scripts are in place. An initial CPU training run has completed one epoch for each task. Its validation results were: abnormality ROC-AUC 0.7846 (accuracy 0.7647, F1 0.8542); ACL ROC-AUC 0.6934 (accuracy 0.6050, F1 0.5766); meniscus ROC-AUC 0.6039 (accuracy 0.6050, F1 0.5347). These are preliminary results from a single epoch and should not be treated as final performance. The next step is to run the fixed evaluation script, inspect the metrics and plots, then plan further training and model improvements.

