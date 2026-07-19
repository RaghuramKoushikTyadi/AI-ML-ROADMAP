# AI/ML Roadmap

Hi, welcome to my AI/ML roadmap repository.

This repository contains my complete learning journey as I study Artificial Intelligence and Machine Learning. I will keep updating it as I learn new topics, build projects, and improve my understanding.

The goal is to build a strong foundation by learning the concepts, implementing them in Python, and applying them to real datasets.

---

## Progress

### Day 1 - Linear Regression

**Topics Covered**

* Introduction to Linear Regression
* Exploratory Data Analysis (EDA)
* Missing value analysis
* Duplicate value analysis
* Correlation analysis
* One-Hot Encoding
* Data preprocessing

**Project**

* Student Performance Prediction

**Libraries Used**

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

**Status**

* Completed data preprocessing
* Completed model training and evaluation

---

### Day 2 - Ridge and Lasso Regression

**Topics Covered**

* Feature Scaling using StandardScaler
* Ridge Regression (L2 Regularization)
* Lasso Regression (L1 Regularization)
* Understanding Regularization
* Alpha (Regularization Strength)
* Feature Selection using Lasso
* Model comparison (Linear vs Ridge vs Lasso)

**Project**

* Comparison of Linear Regression, Ridge Regression, and Lasso Regression on the Student Performance dataset

**Libraries Used**

* Pandas
* NumPy
* Scikit-learn

**Status**

* Implemented Ridge Regression
* Implemented Lasso Regression
* Compared all three regression models
* Experimented with different alpha values
* Completed model evaluation and comparison

---

### Day 3 - Cross Validation and Manual Hyperparameter Tuning

**Topics Covered**

* K-Fold Cross Validation
* Manual Cross Validation
* Model Evaluation using Cross Validation
* Hyperparameter Tuning
* Ridge Alpha Tuning
* Lasso Alpha Tuning

**Project**

* Manual implementation of 5-Fold Cross Validation
* Manual hyperparameter tuning for Ridge and Lasso Regression
* Comparison of multiple alpha values using K-Fold Cross Validation

**Libraries Used**

* Pandas
* NumPy
* Scikit-learn

**Status**

* Implemented manual K-Fold Cross Validation
* Compared Linear Regression, Ridge Regression, and Lasso Regression on identical folds
* Calculated average R² scores across all folds
* Tuned Ridge and Lasso using multiple alpha values
* Observed Ridge remained relatively stable across different alpha values
* Found the best-performing Lasso alpha (approximately 0.15) for this dataset

---

### Day 4 - Scikit-learn Cross Validation and GridSearchCV

**Topics Covered**

* `cross_val_score()`
* `Pipeline`
* Data Leakage
* GridSearchCV
* Automated Hyperparameter Tuning
* Pipeline Parameter Naming (`ridge__alpha`, `lasso__alpha`)

**Project**

* Automated 5-Fold Cross Validation using `cross_val_score()`
* Hyperparameter tuning using `GridSearchCV`
* Comparison of manual tuning with Scikit-learn implementations

**Libraries Used**

* Pandas
* NumPy
* Scikit-learn

**Status**

* Implemented `cross_val_score()`
* Built Pipelines for Linear, Ridge, and Lasso Regression
* Prevented data leakage using `Pipeline`
* Implemented `GridSearchCV`
* Verified that `GridSearchCV` selected the same best alpha as the manual implementation

---

## Roadmap

* [x] Linear Regression
* [x] Multiple Linear Regression
* [x] Ridge Regression
* [x] Lasso Regression
* [x] K-Fold Cross Validation
* [x] Hyperparameter Tuning
* [x] cross_val_score()
* [x] GridSearchCV
* [ ] Logistic Regression
* [ ] Decision Trees
* [ ] Random Forest
* [ ] Support Vector Machines
* [ ] K-Nearest Neighbors
* [ ] Naive Bayes
* [ ] Clustering
* [ ] Principal Component Analysis (PCA)
* [ ] Neural Networks
* [ ] Deep Learning
* [ ] Natural Language Processing
* [ ] Computer Vision

---

I will continue updating this repository as I progress through my AI/ML learning journey.
