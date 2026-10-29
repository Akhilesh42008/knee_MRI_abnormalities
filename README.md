# Team:-
# Akhilesh Bethireddy (SC25B141)
# Gireesh Bhavani Molkalapalli (SC25B145)
# lokesh Jallu (SC25B150)

# knee_MRI_abnormalities
To detect abnormalities in knee using MRI scans<br>

<p>This is a team machine-learning project focused on classifying knee MRI exams using deep learning and PyTorch, based on the Stanford MRNet knee MRI dataset.</p>
<br>
<p>The project aims to develop models for three classification tasks:</p>

<p>1. Abnormality classification — predict whether a knee MRI exam is labeled abnormal.</p>

<p>2. ACL tear classification — predict whether an exam is labeled as having an ACL tear.</p> 

<p>3. Meniscus tear classification — predict whether an exam is labeled as having a meniscus tear.</p>

<p>The initial implementation will focus on ACL tear classification. Once the data pipeline and baseline model work correctly, the project will expand to the other two tasks.</p> 

<b>Project Goals</b>

<p>1. Explore the MRI files and label tables.</p>

<p>2. Verify the official meaning of the labels.</p>

<p>3. Match MRI exams with the correct labels.</p>

<p>4. Preprocess MRI images using NumPy and PyTorch.</p>

<p>5. Build a baseline convolutional neural network (CNN).</p>

<p>6. Train and evaluate separate classifiers for abnormality, ACL tears, and meniscus tears.</p>

<p>7. Evaluate models on held-out exams and document their limitations.</p>

<p>8. Keep the data pipeline reproducible and respect dataset access and usage conditions.</p>

<b>Project updates</b>

<p>1. Created folders for data, models, and src. Project environment for modules. '.gitignore' for ignoring unnecessary files. requirements.txt for managing dependencies. main.py file for the entry point.</p>

<p>2. Accessing stanford university datasets of knee mri abnormalities and loading them into the project from redivis.</p>

<p>3. Created a script to inspect MRNet training and validation label tables.</p>

<p>4. Standardize label-table columns to 'exam_id' and 'label'.</p>

<p>5. Check missing values and duplicate exam IDs.</p>