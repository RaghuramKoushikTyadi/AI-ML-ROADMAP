# Logistic Regression

## Overview

This folder contains my implementation of Logistic Regression for binary classification using the Breast Cancer Wisconsin dataset.

The project covers the complete machine learning classification workflow, including data exploration, preprocessing, model training, probability prediction, and performance evaluation using classification metrics.

---

## Folder Structure

```text
logistic_regression/
│
├── eda_breast_cancer.py
├── breast_cancer_prediction.py
└── README.md
```

---

# Dataset

## Breast Cancer Wisconsin Dataset

Loaded directly from Scikit-learn.

```python
from sklearn.datasets import load_breast_cancer
```

### Target Variable

```
target
```

- **0** → Malignant
- **1** → Benign

The objective is to classify whether a tumor is malignant or benign based on various cell measurements.

---

# Topics Covered

- Introduction to Classification
- Regression vs Classification
- Binary Classification
- Sigmoid Function
- Probability Prediction
- Decision Threshold
- Exploratory Data Analysis (EDA)
- Train/Test Split
- Feature Scaling using StandardScaler
- Logistic Regression
- Model Prediction
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score

---

# Workflow

## 1. Load Dataset

```python
data = load_breast_cancer()
```

Convert the dataset into a Pandas DataFrame.

---

## 2. Exploratory Data Analysis

Performed:

- Dataset inspection
- Feature information
- Missing value analysis
- Statistical summary
- Class distribution

---

## 3. Feature Selection

Separate the dataset into features and target.

```python
X = df.drop(columns="target")
y = df["target"]
```

---

## 4. Train/Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=18
)
```

- 80% Training Data
- 20% Testing Data

---

## 5. Feature Scaling

```python
scaler = StandardScaler()
```

The scaler is fitted on the training data and applied to both training and testing datasets.

---

## 6. Train Model

```python
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)
```

The model learns the relationship between the input features and the probability of belonging to each class.

---

## 7. Prediction

```python
preds = model.predict(X_test)
```

The trained model predicts the class labels for unseen data.

---

## 8. Model Evaluation

Classification metrics used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# Results

| Metric | Value |
|--------|------:|
| Accuracy | 98.25% |
| Precision | 97.06% |
| Recall | 100% |
| F1-Score | 98.51% |

### Confusion Matrix

```
[[46  2]
 [ 0 66]]
```

### Interpretation

- True Negatives (TN): **46**
- False Positives (FP): **2**
- False Negatives (FN): **0**
- True Positives (TP): **66**

The model correctly classified **112 out of 114** test samples, achieving excellent performance with no false negatives.

---

# What I Learned

Through this project I learned:

- Difference between Regression and Classification
- Binary Classification
- Logistic Regression
- Sigmoid Function
- Probability Prediction
- Decision Threshold
- Train/Test Split
- Feature Scaling
- Classification Workflow
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Model Evaluation

---

# Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

# Future Improvements

- Multiclass Logistic Regression
- One-vs-Rest (OvR) Classification
- Multinomial Logistic Regression
- ROC Curve
- ROC-AUC Score
- Precision-Recall Curve
- Threshold Tuning
- Hyperparameter Tuning using GridSearchCV
- Compare Logistic Regression with Decision Trees and Random Forests