# Linear Regression

## Overview

This folder contains my implementations of Linear Regression and related regression concepts using multiple datasets.

The projects cover the complete regression workflow, from data preprocessing and exploratory data analysis to model evaluation, regularization, cross-validation, and hyperparameter tuning.

---

## Folder Structure

```text
linear_regression/
│
├── preprocessing.py
├── lin_reg.py
├── lin_reg_with_g1_g2.py
├── ridge.py
├── lasso.py
├── k_fold_lin_ridge_lasso.py
├── k_fold_multiple_alphas.py
├── cross_val.py
├── grid_search_cv.py
├── housing_eda.py
├── housing_model.py
├── student-mat.csv
├── student-mat-cleaned.csv
├── housing.csv
├── cleaned_housing.csv
└── README.md
```

---

# Datasets Used

## 1. Student Performance Dataset

Target Variable

```
G3
```

Predicts the student's final grade.

---

## 2. California Housing Dataset

Target Variable

```
median_house_value
```

Predicts the median house value using housing-related features.

---

# Topics Covered

- Exploratory Data Analysis (EDA)
- Missing Value Handling
- Duplicate Value Analysis
- Correlation Analysis
- One-Hot Encoding
- Feature Engineering
- Train/Test Split
- Linear Regression
- Multiple Linear Regression
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- Feature Scaling using StandardScaler
- Model Coefficients
- Model Evaluation
- K-Fold Cross Validation
- Manual Hyperparameter Tuning
- `cross_val_score()`
- Pipelines
- Data Leakage Prevention
- GridSearchCV

---

# Workflow

## 1. Data Preprocessing

- Load dataset
- Explore the dataset
- Handle missing values
- Check duplicate rows
- Correlation analysis
- One-hot encode categorical features
- Save cleaned dataset

---

## 2. Feature Selection

Separate the dataset into:

```python
X = df.drop(columns=[target])
y = df[target]
```

---

## 3. Train/Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=18
)
```

---

## 4. Feature Scaling (when required)

```python
StandardScaler()
```

Used for Ridge and Lasso Regression before model training.

---

## 5. Model Training

Implemented:

- Linear Regression
- Ridge Regression
- Lasso Regression

---

## 6. Model Evaluation

Regression metrics used:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

---

## 7. Cross Validation

Implemented both:

- Manual 5-Fold Cross Validation
- `cross_val_score()`

Compared model performance across folds.

---

## 8. Hyperparameter Tuning

Implemented:

- Manual alpha tuning
- `GridSearchCV`

Compared the best alpha values selected by both approaches.

---

# Projects

## Student Performance Prediction

Built multiple regression models to predict students' final grades.

Compared:

- Linear Regression
- Ridge Regression
- Lasso Regression

Also compared performance with and without previous exam scores (G1 and G2).

---

## California Housing Price Prediction

Performed:

- Exploratory Data Analysis
- Data Cleaning
- One-Hot Encoding
- Linear Regression

Predicted median house prices using housing features.

---

# What I Learned

Through these projects I learned:

- Machine Learning workflow
- Feature Engineering
- Regression Modeling
- Feature Scaling
- Regularization
- Ridge Regression
- Lasso Regression
- Hyperparameter Tuning
- K-Fold Cross Validation
- `cross_val_score()`
- Pipelines
- Preventing Data Leakage
- GridSearchCV
- Model Evaluation using MAE, MSE and R² Score

---

# Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Future Improvements

- Compare additional regression models
- Save trained models using Joblib
- Build a Streamlit application
- Deploy trained models
- Experiment with larger real-world regression datasets