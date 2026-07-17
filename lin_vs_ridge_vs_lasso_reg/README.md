# Linear vs Ridge vs Lasso Regression

## Overview

This project compares Linear Regression, Ridge Regression, and Lasso Regression using the Student Performance dataset.

The objective is to understand how regularization affects regression models, how feature scaling impacts regularized models, and how different values of alpha influence model performance.

The project also compares all three models to observe the effect of L1 and L2 regularization on prediction accuracy.

---

## Project Structure

```text
lin_vs_ridge_vs_lasso_reg/
│
├── ridge.py
├── lasso.py
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

### 3. Split dataset

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

---

### 4. Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Feature scaling is applied before training Ridge and Lasso Regression so that every feature contributes on the same scale during regularization.

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

Applies L2 regularization, shrinking coefficients while keeping every feature.

---

#### Lasso Regression

```python
model = Lasso(alpha=0.1)
```

Applies L1 regularization, shrinking coefficients and automatically removing less important features by setting some coefficients to zero.

---

### 6. Prediction

```python
preds = model.predict(X_test)
```

Each trained model predicts the final grades for unseen students.

---

### 7. Evaluation

Metrics used

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

The learned coefficients were also compared to observe the effect of regularization.

---

# Results

| Model                      |  MAE |  MSE |   R² |
| -------------------------- | ---: | ---: | ---: |
| Linear Regression          | 1.25 | 2.79 | 0.82 |
| Ridge Regression           | 1.25 | 2.79 | 0.82 |
| Lasso Regression (α = 0.1) | 1.02 | 2.15 | 0.86 |

### Observation

Linear Regression provided a strong baseline for prediction.

Ridge Regression produced almost identical results, indicating that the dataset did not suffer from significant overfitting.

Lasso Regression achieved the best performance after tuning the alpha value. By removing weaker features while keeping the important ones, it improved generalization on this dataset.

---

# What I Learned

Through this project I learned:

* Why StandardScaler is important for Ridge and Lasso Regression.
* Difference between L1 and L2 regularization.
* Ridge shrinks coefficients but keeps every feature.
* Lasso can shrink coefficients to exactly zero, performing automatic feature selection.
* Alpha controls the strength of regularization.
* Smaller alpha does not always produce better results; the objective is to find the best balance between model complexity and generalization.
* Different models respond differently to the same alpha value.
* Regularization is used to improve generalization rather than simply increase training accuracy.

---

# Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

# Future Improvements

* Implement K-Fold Cross Validation
* Compare average model performance across multiple folds
* Tune alpha using GridSearchCV
* Compare additional regression algorithms
