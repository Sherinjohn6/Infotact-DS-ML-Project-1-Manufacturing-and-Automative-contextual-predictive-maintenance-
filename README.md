# 🔧 Predictive Maintenance System (AI4I 2020 Dataset)

## 📌 Overview

This project presents an end-to-end **Machine Learning solution for Predictive Maintenance** using the **AI4I 2020 dataset**.  
The system predicts **machine failure events** based on industrial sensor data to help reduce downtime, improve safety, and optimize maintenance cost.

The solution includes:
- Data preprocessing & feature engineering  
- Handling class imbalance using SMOTE  
- Model training with LightGBM  
- Hyperparameter optimization  
- Robustness validation  
- Model explainability using SHAP  
- Production-ready deployment using Streamlit  

---

## 🎯 Objective

To build a reliable classification model that predicts:

> **Machine Failure (Yes / No)** based on real-time sensor readings.

---

## 📊 Dataset Description

**Source:** UCI Machine Learning Repository  
**Type:** Industrial predictive maintenance dataset  
**Samples:** 10,000 records  
**Task:** Binary Classification  

### Features:
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (rpm)
- Torque (Nm)
- Tool Wear (min)
- Product Type (L / M / H)

### Target:
- `Machine failure` → 0 (No Failure), 1 (Failure)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Encoding categorical variable (`Product Type`)
- Feature scaling using Standard/Robust Scaler
- Train-test split with stratification

---

### 2. Handling Class Imbalance (SMOTE)

The dataset is highly imbalanced with rare failure cases.

SMOTE (Synthetic Minority Oversampling Technique) was applied to:
- Balance class distribution  
- Improve recall for failure detection  
- Reduce model bias toward majority class  

---

### 3. Feature Engineering
- Temperature difference features
- Operational stress indicators
- Tool wear intensity mapping
- Feature selection to remove redundancy

---

### 4. Model Development
Primary model:
- **LightGBM Classifier**

Baseline comparisons:
- Logistic Regression  
- Random Forest  
- XGBoost  

---

### 5. Hyperparameter Tuning
Optimized using GridSearchCV / RandomizedSearchCV.

Key parameters tuned:
- num_leaves  
- max_depth  
- learning_rate  
- n_estimators  
- subsample  
- min_child_samples  

---

### 6. Robustness Evaluation
Model stability validated using:
- K-Fold Cross Validation  
- Multiple random seed testing  
- Noise injection experiments  
- Train-test split variations  

Result: Stable performance with low variance across runs.

---

### 7. Model Evaluation Metrics
- Accuracy  
- Precision  
- Recall (critical metric)  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

---

### 8. Model Explainability (SHAP)
SHAP analysis was used to interpret predictions.

Key insights:
- Torque is the strongest failure indicator  
- Tool wear significantly influences failure risk  
- Temperature imbalance contributes to system stress  
- Rotational speed affects operational stability  

---

## 🚀 Deployment

The model is deployed as an interactive web application using **Streamlit**.

### Deployment Features:
- User input for sensor values  
- Real-time prediction  
- Failure probability output  
- Clean and interactive UI  

### Deployment Stack:
- Streamlit  
- Pickle (Model Serialization)  
- Scikit-learn Pipeline  

### Deployment Options:
- Streamlit Cloud  
- Render / Railway  
- AWS EC2 / Azure App Service  
- Docker-based deployment  

---

## 🧱 Project Structure

```bash
project/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── feature_names.pkl
│
├── data_preprocessing.py
├── feature_engineering.py
├── smote_pipeline.py
├── hyperparameter_tuning.py
├── shap_analysis.py
│
├── X_train.pkl
├── X_test.pkl
│
└── README.md
