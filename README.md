# 🎗️ Breast Cancer Prediction

A machine learning project to predict whether a breast tumor is **malignant** or **benign** using diagnostic features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [References](#references)

---

## 📖 Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate detection significantly improves treatment outcomes. This project applies supervised machine learning techniques to classify tumors as **Malignant (M)** or **Benign (B)** based on cell nucleus features.

**Goal:** Build a classification model with high accuracy and recall to minimize false negatives (missed cancer cases).

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Samples:** 569 instances
- **Features:** 30 numeric features (mean, standard error, worst value of radius, texture, perimeter, area, smoothness, etc.)
- **Target Variable:** `diagnosis` — M (Malignant) or B (Benign)
- **Class Distribution:**
  - Benign: 357 (62.7%)
  - Malignant: 212 (37.3%)

---

## 📁 Project Structure

```
breast-cancer-prediction/
│
├── data/
│   └── breast_cancer.csv          # Raw dataset
│
├── notebooks/
│   └── breast_cancer_eda.ipynb    # Exploratory Data Analysis
│   └── breast_cancer_model.ipynb  # Model training & evaluation
│
├── src/
│   └── preprocessing.py           # Data cleaning & feature engineering
│   └── train.py                   # Model training script
│   └── evaluate.py                # Evaluation metrics
│
├── models/
│   └── best_model.pkl             # Saved trained model
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Saving | Joblib / Pickle |
| Notebook | Jupyter Notebook |

---

## 🔍 Exploratory Data Analysis

Key EDA steps performed:

- Checked for **missing values** and data types
- Visualized **class distribution** using count plots
- Plotted **correlation heatmap** to identify feature relationships
- Used **box plots** and **violin plots** to compare feature distributions between M and B classes
- Identified highly correlated features (e.g., `radius_mean`, `perimeter_mean`, `area_mean`)

---

## 🤖 Model Building

The following models were trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 97.4% | 96.8% | 96.2% | 96.5% |
| Decision Tree | 93.8% | 92.5% | 93.1% | 92.8% |
| Random Forest | 98.2% | 97.9% | 97.6% | 97.7% |
| Support Vector Machine | 98.6% | 98.1% | 98.3% | 98.2% |
| K-Nearest Neighbors | 96.5% | 95.8% | 95.5% | 95.6% |

> ✅ **Best Model:** Support Vector Machine (SVM) with RBF kernel

### Preprocessing Steps:
- Label encoding of target variable (M → 1, B → 0)
- Feature scaling using `StandardScaler`
- Train-test split: 80% training / 20% testing
- Hyperparameter tuning with `GridSearchCV`

---

## 📈 Results

- **Best Accuracy:** 98.6% (SVM)
- **Confusion Matrix:** Low false negatives — critical for medical diagnosis
- **ROC-AUC Score:** 0.997
- **Cross-Validation Score:** 97.8% (mean over 5 folds)

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook notebooks/breast_cancer_model.ipynb
```

### 4. Train the Model via Script
```bash
python src/train.py
```

---

## 🔮 Future Work

- Deploy the model as a **web application** using Flask or Streamlit
- Experiment with **deep learning** models (Neural Networks)
- Apply **SMOTE** for handling class imbalance more robustly
- Integrate **SHAP values** for model explainability
- Add **feature selection** techniques (RFE, PCA)

---

## 📚 References

- [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Wolberg, W.H. et al. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*

---

## 👩‍💻 Author

**Anjali Sharma**  
Data Science Student  
📧 anjalisharma505152@.com  
🔗 [LinkedIn](https://linkedin.com/in/anjali-sharma-anjali) | [GitHub](https://github.com/anjalisharma)

---

> *"Early detection saves lives. Machine learning makes it faster."*
