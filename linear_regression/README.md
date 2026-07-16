# Student Performance Prediction using Linear Regression

## Overview

This project predicts a student's final grade (G3) using Linear Regression.

The project follows the complete machine learning workflow:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Performance comparison

Two different models were built to compare how feature selection affects prediction accuracy.

---

## Project Structure

```
linear_regression/
│
├── preprocessing.py
├── lin_reg.py
├── lin_reg_with_g1_g2.py
├── student-mat.csv
├── student-mat-cleaned.csv
└── README.md
```

---

## Dataset

Dataset used:

Student Performance Dataset

Target Variable

```
G3
```

Final student grade.

---

## Preprocessing

The preprocessing script performs the following steps.

- Load dataset
- Inspect dataset
- Check missing values
- Check duplicate rows
- Create correlation heatmap
- One-hot encode categorical columns
- Save cleaned dataset

Output

```
student-mat-cleaned.csv
```

---

## Model Workflow

### 1. Load cleaned dataset

```python
df = pd.read_csv(path)
```

---

### 2. Select features and target

### Model 1

Previous grades removed.

```python
X = df.drop(columns=["G1","G2","G3"])
Y = df["G3"]
```

This model predicts the final grade without using previous exam scores.

---

### Model 2

Previous grades included.

```python
X = df.drop(columns="G3")
Y = df["G3"]
```

This model predicts the final grade using every available feature.

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

- 80% Training Data
- 20% Testing Data

---

### 4. Train Model

```python
model = LinearRegression()

model.fit(X_train,Y_train)
```

The model learns the relationship between the input features and the target variable.

---

### 5. Learned Parameters

```python
model.coef_
```

Returns the learned weight for every feature.

Positive coefficient

- Increases predicted grade

Negative coefficient

- Decreases predicted grade

---

```python
model.intercept_
```

Returns the starting prediction before feature contributions are added.

---

### 6. Prediction

```python
preds = model.predict(X_test)
```

The trained model predicts grades for students it has never seen before.

---

### 7. Evaluation

Metrics used

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

---

# Results

## Model 1 (Without G1 and G2)

| Metric | Value |
|--------|------:|
| MAE | 2.70 |
| MSE | 12.68 |
| R² | 0.18 |

### Observation

The model predicts the final grade with an average error of about **2.7 marks**.

Since previous grades were removed, the model relies only on demographic, family and study related features.

This makes prediction much harder.

---

## Model 2 (With G1 and G2)

| Metric | Value |
|--------|------:|
| MAE | 1.25 |
| MSE | 2.79 |
| R² | 0.82 |

### Observation

Including G1 and G2 significantly improves prediction performance.

Average prediction error drops to around **1.25 marks**, and the model explains about **82%** of the variation in final grades.

This shows how strongly previous academic performance influences the final grade.

---

# What I Learned

Through this project I learned:

- Machine Learning workflow
- Difference between features (X) and target (Y)
- Train/Test Split
- Data leakage
- Linear Regression
- Model coefficients
- Model intercept
- Model prediction
- MAE
- MSE
- R² Score
- Feature selection
- Effect of different features on model performance

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

- Build a Streamlit application
- Save the trained model using Joblib
- Compare Linear Regression with other regression models
- Hyperparameter tuning
- Deploy the project