# ✅ Machine Learning Topics Checklist

Use this checklist to track your progress through the full ML journey — from foundations to deployment-ready models.

---

## 🟢 1. Core Foundations (Math + Tools)

- [x] Linear Algebra: Vectors, Matrices, Dot Products, Inverses
- [x] Probability & Statistics: Bayes' Rule, Expectation, Variance, Distributions
- [x] Calculus: Derivatives, Gradients, Chain Rule
- [x] Python Basics + Numpy
- [x] `pandas`, `matplotlib`, `cupy`, `cudf` basics

---

## 🟡 2. Working with Data

### 🛠️ Preprocessing & Feature Engineering
- [x] Handling Missing Data ✅
- [x] Categorical Encoding (One-Hot, Label) ✅
- [x] Feature Scaling (Z-score, IQR, Log) ✅
- [ ] Feature Selection (Variance Threshold, Correlation)
- [ ] Outlier Detection & Handling
- [x] Skewness & Normalization ✅
- [ ] Handling Class Imbalance (SMOTE, Undersampling)

---

## 🧪 3. Model Evaluation & Metrics

- [x] Mean Squared Error (MSE) ✅
- [x] Root Mean Squared Error (RMSE) ✅
- [x] Mean Absolute Error (MAE) ✅
- [x] R² and Adjusted R² ✅
- [ ] Accuracy, Precision, Recall, F1-score
- [ ] Confusion Matrix
- [ ] ROC-AUC Curve, PR Curve
- [ ] Log Loss
- [ ] Kappa Score (optional)

---

## 🔵 4. Supervised Learning

### 🔹 Regression
- [x] Linear Regression (Normal Equation)
- [x] Linear Regression (Gradient Descent)
- [ ] Polynomial Regression ✅
- [ ] Ridge & Lasso Regression (L1/L2 Regularization)
- [x] Logistic Regression (Binary Classification) ✅
- [ ] Multinomial Logistic Regression

### 🔹 Classification
- [ ] K-Nearest Neighbors (KNN)
- [ ] Decision Trees
- [ ] Random Forests
- [ ] Naive Bayes
- [ ] Support Vector Machines (SVM)

---

## 🟣 5. Model Selection & Hyperparameter Tuning

- [ ] Bias-Variance Tradeoff
- [ ] Cross-Validation (K-Fold, Stratified K-Fold) ✅ `logistic_regression.py`
- [x] Grid Search ✅ `cuml_linear_regression.py`
- [ ] Random Search
- [ ] Early Stopping (for iterative models)

---

## 🟤 6. Unsupervised Learning (Expanded)

### 📊 Clustering
- [ ] K-Means Clustering
- [ ] Hierarchical Clustering (Agglomerative & Divisive)
- [ ] DBSCAN
- [ ] Gaussian Mixture Models (GMM)
- [ ] Spectral Clustering (optional)

### 📐 Dimensionality Reduction
- [ ] Principal Component Analysis (PCA)
- [ ] t-SNE (T-distributed Stochastic Neighbor Embedding)
- [ ] UMAP (Uniform Manifold Approximation)
- [ ] Autoencoders (Intro)

### 🔍 Association Rule Learning
- [ ] Apriori Algorithm
- [ ] Eclat Algorithm
- [ ] FP-Growth Algorithm

### ⚠️ Anomaly/Outlier Detection
- [ ] Z-Score / IQR Method (basic stats-based)
- [ ] Isolation Forest
- [ ] One-Class SVM
- [ ] DBSCAN (used for outlier detection too)

### ⏳ Time Series (Optional Intro)
- [ ] Time Series Decomposition (Trend, Seasonality)
- [ ] Clustering Time Series (Dynamic Time Warping)

---

## 🔴 7. Ensemble Methods

- [ ] Bagging (e.g., Random Forests revisit)
- [ ] Boosting (AdaBoost, Gradient Boosting)
- [ ] XGBoost
- [ ] LightGBM
- [ ] Stacking & Blending

---

## ⚫ 8. Deployment, Pipelines, and Extras

- [x] Saving/Loading Models (`joblib`, `pickle`) ✅ used in all models
- [ ] Building ML Pipelines (e.g., `Pipeline` API) ✅ `cuml_linear_regression.py`
- [ ] Real-world Datasets: Iris, Titanic, Diabetes ✅ `config.py` supports
- [ ] Basic CI/CD for ML (optional)
- [ ] Introduction to DVC (Data Version Control)
- [ ] Model Monitoring Concepts

---

## 🧠 9. Interpretability (Optional but Important)

- [ ] Feature Importance (for Trees & Linear models)
- [ ] SHAP (SHapley Additive explanations)
- [ ] LIME (Local Interpretable Model-agnostic Explanations)

---

📌 *Check `[ ]` to `[x]` as you progress. All ✅ symbols show where your current codebase already covers key topics.*
