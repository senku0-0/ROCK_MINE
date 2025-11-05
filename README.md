# ROCK_MINE — Sonar Rocks vs Mines Classification

Project summary
This project implements a binary classification model using Logistic Regression to distinguish between rocks and mines from sonar signal data. The aim was to practice a full supervised learning workflow: data inspection, preprocessing, train/test split, model training, evaluation, and inference on custom inputs. The dataset used is the classic "sonar" dataset (CSV: `sonar data.csv`) where the label column contains two classes: "R" (rock) and "M" (mine).

What I learned
- How to perform an end-to-end binary classification workflow in Python using scikit-learn.
- How to inspect and validate a dataset (data shape, missing values, basic statistics).
- Why and how to use train/test splitting with stratification to preserve class proportions.
- How to train and evaluate a Logistic Regression classifier for binary tasks.
- How to make and format a custom prediction from raw feature values.
- Best practices for reproducibility by setting random seeds (random_state).
- How to structure code for clarity (separate data loading, training, evaluation, and inference steps).

What I used
- Python (pandas, numpy)
- scikit-learn (train_test_split, LogisticRegression, accuracy_score)
- Dataset file: `sonar data.csv` (no header, 207 rows × 61 columns)
- Development environment: any Python 3.7+ environment; recommended to use a virtual environment

Project structure (recommended)
- sonar_data.csv or sonar data.csv — dataset (CSV, target as final column)
- train_model.py or notebook.ipynb — training and evaluation script / notebook
- predict.py — inference script for single custom inputs
- README.md — this file

Notes on the dataset
- The CSV has 61 columns with no header; columns 0–59 are numeric features and column 60 is the label.
- Labels: "R" → Rock, "M" → Mine.
- The provided code shows that there are no missing values and the dataset size is small (207 records), so models must be evaluated carefully for overfitting.

Methodology / Workflow
1. Load data
   - Read `sonar data.csv` with pandas using `header=None`.
   - Convert to DataFrame and inspect with `head()`, `shape`, `isnull().sum()`, and `describe()`.

2. Prepare features and labels
   - X = all columns except the last (columns 0–59)
   - Y = column 60 (target labels "R"/"M")

3. Train / test split
   - Use `train_test_split` with `test_size=0.2`, `stratify=Y` (to maintain label proportions), and `random_state=1` for reproducibility.

4. Train model
   - Instantiate `LogisticRegression()` and call `fit(X_train, Y_train)`.

5. Evaluate
   - Use `predict()` on training and test sets and compute `accuracy_score`.
   - Print training and testing accuracy to check for under/overfitting.

6. Inference on custom input
   - Prepare a 1×60 array for a new sample, reshape via `.reshape(1, -1)` and call `model.predict(...)`.
   - Interpret the returned label ("R" or "M").

How to run (example)
- From a script (recommended file name: `train_model.py`):
  1. Ensure Python 3.7+ is installed and a virtual environment is active.
  2. Install dependencies:
     pip install -r requirements.txt
     (requirements could contain: pandas, numpy, scikit-learn)
  3. Place `sonar data.csv` in the project root.
  4. Run the training script:
     python train_model.py
  5. The script prints dataset info, training accuracy, testing accuracy, and a sample custom prediction.

Example of the inference flow (as shown in the code)
- Create an input numpy array of length 60 with float feature values.
- Reshape the array to (1, -1) and call `model.predict(...)`.
- The prediction returns a label string: "R" or "M".

Design rationale
- Logistic Regression is an appropriate baseline for binary classification problems: it is fast, interpretable, and often works well on smaller datasets.
- Stratified train/test split preserves label distribution and reduces evaluation bias.
- Explicit reshaping of inputs before prediction prevents dimensionality errors and makes the inference pipeline robust.

Limitations & next steps
- Small dataset size (207 samples): evaluate model stability using cross-validation and report mean/std of metrics.
- Explore feature scaling (StandardScaler) and regularization strength (C parameter) to improve model performance.
- Add confusion matrix, precision/recall/F1 metrics — accuracy alone can be misleading for imbalanced data.
- Save / load model artifacts using joblib/pickle for reproducible deployment and inference.
- Refactor code into functions or modules and add unit tests and a requirements.txt for reproducible environments.
- Consider experiments with more complex models (SVM, Random Forest) and model selection via GridSearchCV.

