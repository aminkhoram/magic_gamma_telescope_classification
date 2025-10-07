MAGIC Gamma Classification:

Binary classification of MAGIC Gamma Telescope events (gamma vs hadron) using multiple models: K-NN, Naive Bayes, Logistic Regression, SVM, and a TensorFlow MLP. Includes EDA histograms, train/validation/test split, feature scaling, optional oversampling to handle class imbalance, and model evaluation with classification_report.

Dataset:

File: magic04.data (CSV, no header)
https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

Columns:

fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist, class


Target encoding: "g" → 1 (gamma), "h" → 0 (hadron)

Place magic04.data in the project root (same folder as the script/notebook).

Environment & Dependencies

Python 3.10+ recommended

Install dependencies:

pip install numpy 
pip install pandas 
pip install matplotlib 
pip install scikit-learn 
pip install imbalanced-learn 
pip install tensorflow


Optional (for notebooks):

pip install jupyter

Quick Start:

Load & preprocess

Reads magic04.data, assigns column names.

Converts target to binary.

Plots feature histograms per class.

Split:

Shuffles the dataset and splits into:

Train: 60%

Validation: 20%

Test: 20%

Scale & (optionally) oversample

Standardizes features with StandardScaler.

Apply RandomOverSampler on the training set (It is used to balance the classes in your dataset by randomly duplicating samples from the minority class until all classes have the same number of samples).

Train models:

K-NN (n_neighbors=5)

Gaussian Naive Bayes

Logistic Regression

SVM (RBF)

TensorFlow MLP (grid over hidden units, dropout, learning rate, batch size)

Evaluate

Prints classification_report (precision, recall, f1, support) on the test set.

For the MLP, picks the configuration with lowest validation loss and evaluates it on the test set.


Output:

Per-feature histograms comparing gamma vs hadron distributions.

Text reports for each classical model.

Training curves (loss/accuracy) for MLP setting.

Final test-set report for the best MLP.

Project Structure (suggested)

Reproducibility:

Set seeds for NumPy / TensorFlow and use random_state in splitters/samplers.

Persist the scaler fitted on the training set if you need to reuse it.

Notes on Imbalance

Oversampling is applied only to the training set to avoid leaking synthetic information into validation/test.
