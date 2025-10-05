MAGIC Gamma Classification

Binary classification of MAGIC Gamma Telescope events (gamma vs hadron) using multiple models: K-NN, Naive Bayes, Logistic Regression, SVM, and a small TensorFlow MLP. Includes EDA histograms, train/validation/test split, feature scaling, optional oversampling to handle class imbalance, and model evaluation with classification_report.

Dataset

File: magic04.data (CSV, no header)

Columns:

fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist, class


Target encoding: "g" → 1 (gamma), "h" → 0 (hadron)

Place magic04.data in the project root (same folder as the script/notebook).

Environment & Dependencies

Python 3.10+ recommended

Install dependencies:

pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow


Optional (for notebooks):

pip install jupyter

Quick Start

Load & preprocess

Reads magic04.data, assigns column names.

Converts target to binary.

Plots feature histograms per class.

Split

Shuffles the dataset and splits into:

Train: 60%

Validation: 20%

Test: 20%

Scale & (optionally) oversample

Standardizes features with StandardScaler.

Applies RandomOverSampler on the training set (optional flag).

Train models

K-NN (n_neighbors=5)

Gaussian Naive Bayes

Logistic Regression

SVM (RBF)

TensorFlow MLP (grid over hidden units, dropout, learning rate, batch size)

Evaluate

Prints classification_report (precision, recall, f1, support) on the test set.

For the MLP, picks the configuration with lowest validation loss and evaluates it on the test set.

Running

If using a script (e.g., main.py), simply:

python main.py


If using a notebook, run cells top-to-bottom.

Output

Per-feature histograms comparing gamma vs hadron distributions.

Text reports for each classical model.

Training curves (loss/accuracy) for each MLP setting.

Final test-set report for the best MLP.

Project Structure (suggested)
.
├── magic04.data
├── README.md
├── requirements.txt
└── src/
    ├── data.py           # load_data(), train_valid_test_split(...)
    ├── preprocess.py     # fit_scaler(), transform(), oversample_train(...)
    ├── models.py         # build/train classical models and MLP
    ├── train.py          # orchestrates workflow
    └── viz.py            # histogram plotting, history plotting

Reproducibility

Set seeds for NumPy / TensorFlow and use random_state in splitters/samplers.

Persist the scaler fitted on the training set if you need to reuse it.

Notes on Imbalance

Oversampling is applied only to the training set to avoid leaking synthetic information into validation/test.
