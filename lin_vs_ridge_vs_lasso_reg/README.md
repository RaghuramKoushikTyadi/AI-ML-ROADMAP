# Linear vs Ridge vs Lasso Regression

## Overview

This project compares Linear Regression, Ridge Regression, and Lasso Regression using the Student Performance dataset.

The objective is to understand how regularization affects regression models, how feature scaling impacts regularized models, how K-Fold Cross Validation provides a more reliable evaluation, and how different values of alpha influence model performance.

The project compares all three models using manual K-Fold Cross Validation, `cross_val_score()`, and `GridSearchCV` to identify the best alpha value for Ridge and Lasso Regression.

---

## Project Structure

```text
lin_vs_ridge_vs_lasso_reg/
│
├── ridge.py
├── lasso.py
├── lin_reg_with_g1_g2.py
├── k_fold_lin_ridge_lasso.py
├── k_fold_multiple_alphas.py
├── cross_val_score.py
├── gridcv.py
├── student-mat-cleaned.csv
└── README.md
```

---

## Dataset

Dataset used:

Student Performance Dataset

### Target Variable

```text
G3
```

Final student grade.

---

## Model Workflow

### 1. Load cleaned dataset

```python
df = pd.read_csv(path)
```

The cleaned dataset generated during the preprocessing stage is loaded.

---

### 2. Select features and target

```python
X = df.drop(columns="G3")
Y = df["G3"]
```

All available features are used to predict the final student grade.

---

### 3. Train/Test Split

```python
train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=18
)
```

* 80% Training Data
* 20% Testing Data

Used for the initial comparison of Linear, Ridge, and Lasso Regression.

---

### 4. Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Feature scaling is applied before training Ridge and Lasso Regression so that every feature contributes equally during regularization.

For K-Fold Cross Validation, scaling is performed inside a `Pipeline`, ensuring that the scaler is fitted separately within each training fold and preventing data leakage.

---

### 5. Train Models

#### Linear Regression

```python
model = LinearRegression()
```

Used as the baseline model without regularization.

---

#### Ridge Regression

```python
model = Ridge(alpha=0.1)
```

Applies L2 regularization by shrinking coefficients while keeping every feature.

---

#### Lasso Regression

```python
model = Lasso(alpha=0.15)
```

Applies L1 regularization by shrinking coefficients and removing less important features.

---

### 6. Manual K-Fold Cross Validation

```python
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=18
)
```

The dataset is split into five folds.

For every fold:

* Training data is selected.
* Feature scaling is applied.
* Linear Regression, Ridge Regression, and Lasso Regression are trained.
* R² score is calculated.
* The average R² score is computed across all folds.

---

### 7. cross_val_score()

Scikit-learn's `cross_val_score()` was used to perform the same 5-Fold Cross Validation automatically.

A `Pipeline` containing `StandardScaler` and each regression model was used to ensure feature scaling was performed independently inside every fold, preventing data leakage.

The mean and standard deviation of the cross-validation scores were used to compare model performance.

---

### 8. Manual Hyperparameter Tuning

Multiple alpha values were tested manually for both Ridge and Lasso Regression.

Example:

```python
alphas=[
    0.025,
    0.05,
    0.075,
    0.1,
    0.125,
    0.15
]
```

Each alpha was evaluated using 5-Fold Cross Validation to determine the best average R² score.

---

### 9. GridSearchCV

`GridSearchCV` was used to automate hyperparameter tuning for Ridge and Lasso Regression.

Example:

```python
ridge_para={"ridge__alpha":[0.0001,0.001,0.01,0.1,0.15]}
lasso_para={"lasso__alpha":[0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.3]}
```

Using 5-Fold Cross Validation, `GridSearchCV` evaluated every alpha value and automatically selected the model with the highest average R² score.

The best alpha obtained matched the manually tuned result.

---

# Results

### Single Train/Test Split

| Model                      |   R² |
| -------------------------- | ---: |
| Linear Regression          | 0.82 |
| Ridge Regression           | 0.82 |
| Lasso Regression (α = 0.1) | 0.86 |

---

### Manual 5-Fold Cross Validation

| Model                            | Average R² |
| -------------------------------- | ---------: |
| Linear Regression                |     0.8074 |
| Ridge Regression                 |     0.8074 |
| Lasso Regression (Best α ≈ 0.15) | **0.8234** |

---

### GridSearchCV

| Model            | Best Alpha | Best Average R² |
| ---------------- | ---------: | --------------: |
| Ridge Regression |       0.15 |          0.7843 |
| Lasso Regression |       0.15 |      **0.8116** |

---

### Observation

Linear Regression provided a strong baseline.

Ridge Regression produced nearly identical performance across different alpha values, indicating that this dataset is not highly sensitive to L2 regularization.

Lasso Regression consistently produced the best average R² score after hyperparameter tuning.

Both manual tuning and `GridSearchCV` selected **0.15** as the best alpha, validating the manual implementation.

Using a `Pipeline` ensured preprocessing occurred independently within every fold, preventing data leakage during cross-validation.

---

# What I Learned

Through this project I learned:

* Why `StandardScaler` is important for Ridge and Lasso Regression.
* Difference between L1 and L2 regularization.
* Ridge shrinks coefficients but keeps every feature.
* Lasso can shrink coefficients to exactly zero, performing automatic feature selection.
* Alpha controls the strength of regularization.
* Different models respond differently to the same alpha value.
* How manual K-Fold Cross Validation works.
* How `cross_val_score()` automates K-Fold Cross Validation.
* Why preprocessing must be performed inside each fold to avoid data leakage.
* How `Pipeline` prevents data leakage by fitting the scaler only on the training portion of each fold.
* Why the average cross-validation score is more reliable than a single train/test split.
* How manual hyperparameter tuning works.
* How `GridSearchCV` automates hyperparameter tuning and selects the best-performing model.

---

# Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

# Future Improvements

* Implement `RandomizedSearchCV`
* Explore Elastic Net Regression
* Compare additional regression algorithms
* Apply these techniques to larger real-world datasets
