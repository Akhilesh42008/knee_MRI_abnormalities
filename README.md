# Team Members

- **Akhilesh Bethireddy (SC25B141)**
- **Gireesh Bhavani Molkalapalli (SC25B145)**
- **Lokesh Jallu (SC25B150)**

# knee_MRI_abnormalities

**To detect abnormalities in the knee using MRI scans.**

<br>

<p>This is a team-based machine learning project focused on classifying knee MRI examinations using deep learning techniques and PyTorch, based on the Stanford MRNet Knee MRI dataset.</p>

<br>

<p>The project aims to develop models for the following three classification tasks:</p>

<p>1. <b>Abnormality Classification</b> — Predict whether a knee MRI examination is labeled as abnormal.</p>

<p>2. <b>ACL Tear Classification</b> — Predict whether an examination is labeled as having an Anterior Cruciate Ligament (ACL) tear.</p>

<p>3. <b>Meniscus Tear Classification</b> — Predict whether an examination is labeled as having a meniscus tear.</p>

<p>The initial implementation will focus on ACL tear classification. Once the data pipeline and baseline model are developed and tested successfully, the project will be extended to the other two classification tasks.</p>

<br>

<b>Project Goals</b>

<p>1. Explore and understand the MRI files and corresponding label tables.</p>

<p>2. Verify and understand the meaning of the labels provided in the dataset.</p>

<p>3. Match MRI examinations with their corresponding labels using examination IDs.</p>

<p>4. Preprocess MRI images using NumPy and PyTorch.</p>

<p>5. Build a baseline Convolutional Neural Network (CNN) for classification.</p>

<p>6. Train and evaluate separate classifiers for abnormality detection, ACL tears, and meniscus tears.</p>

<p>7. Evaluate the models using held-out examinations and document their performance and limitations.</p>

<p>8. Maintain a reproducible data-processing pipeline while respecting dataset access and usage conditions.</p>

<br>

<b>Project Updates</b>

<p>1. Created the initial project structure with separate folders for data, models, and source code (src). Set up the Python project environment, created a .gitignore file to exclude unnecessary files, added requirements.txt for dependency management, and created main.py as the project entry point.</p>

<p>2. Established a connection to the Stanford MRNet Knee MRI dataset through Redivis to access the available dataset tables.</p>

<p>3. Created Python scripts to inspect the MRNet training and validation label tables for all three classification tasks.</p>

<p>4. Standardized the label-table column names to <code>exam_id</code> and <code>label</code> for consistent data processing.</p>

<p>5. Implemented checks for missing values and duplicate examination IDs to verify the consistency of the label tables.</p>
