# 🔧 Predictive Maintenance System using Machine Learning (AI4I 2020 Dataset)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Overview

This project presents an end-to-end **Predictive Maintenance System** built using the **AI4I 2020 Predictive Maintenance Dataset**. The objective is to predict machine failures before they occur by analyzing industrial sensor data, enabling proactive maintenance, reducing downtime, and minimizing maintenance costs.

The project follows a complete machine learning pipeline, including data preprocessing, feature engineering, handling class imbalance using **SMOTE**, model comparison, hyperparameter tuning, robustness evaluation, SHAP explainability, and deployment using **Streamlit**.

---

# 🎯 Project Objectives

- Predict machine failures using industrial sensor data.
- Compare multiple machine learning models.
- Improve prediction accuracy using feature engineering.
- Handle imbalanced data using SMOTE.
- Optimize model performance through hyperparameter tuning.
- Interpret predictions using SHAP.
- Deploy the final model as an interactive Streamlit web application.

---

# 📊 Dataset Information

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

**Task:** Binary Classification

**Total Samples:** 10,000

### Input Features

- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (RPM)
- Torque (Nm)
- Tool Wear (Minutes)
- Product Type (L / M / H)
- TWF
- HDF
- OSF
- PWF
- RNF
- UDI
- Product ID

### Target

Machine Failure

- 0 → No Failure
- 1 → Failure

---

# ⚙️ Project Workflow

```text
                     AI4I 2020 Dataset
                              │
                              ▼
                   Data Preprocessing
          • Data Cleaning
          • Label Encoding
          • Feature Scaling
                              │
                              ▼
                Train-Test Split (Stratified)
                              │
                              ▼
                  Handle Class Imbalance
                          (SMOTE)
                              │
                              ▼
                  Feature Engineering
      • Temperature Difference
      • Power (Torque × RPM)
      • Wear Rate
      • Temperature Ratio
      • Torque/RPM Ratio
                              │
                              ▼
                  Model Development
         ┌────────────────────────────┐
         │                            │
         ▼                            ▼
 Random Forest              LightGBM Classifier
         │                            │
         └──────────────┬─────────────┘
                        ▼
            Model Performance Comparison
                        │
                        ▼
           Hyperparameter Optimization
                 (GridSearchCV)
                        │
                        ▼
             Best Model Selection
                 ✅ LightGBM
                        │
                        ▼
              Robustness Validation
       • Cross Validation
       • Noise Injection
       • Random Seed Testing
                        │
                        ▼
              SHAP Explainability
                        │
                        ▼
             Save Trained Model (.pkl)
                        │
                        ▼
             Streamlit Web Application
                        │
                        ▼
         Real-Time Machine Failure Prediction
```

---

# 🛠 Data Preprocessing

The following preprocessing steps were performed:

- Data Cleaning
- Label Encoding for Product Type
- Feature Scaling
- Train-Test Split
- Feature Selection

---

# ⚡ Handling Class Imbalance

The dataset contains significantly fewer machine failure instances than normal operations.

To address this issue:

- SMOTE (Synthetic Minority Oversampling Technique) was applied.
- Balanced class distribution.
- Improved recall for machine failure prediction.
- Reduced bias toward the majority class.

---

# 🧠 Feature Engineering

Several engineered features were created to improve model performance.

Examples include:

- Temperature Difference
- Power (Torque × RPM)
- Wear Rate
- Temperature Ratio
- Torque-to-RPM Ratio

These features provide additional information about machine operating conditions.

---

# 🤖 Machine Learning Models

Two classification models were trained and compared.

| Model | Purpose |
|--------|----------|
| Random Forest Classifier | Baseline Model |
| LightGBM Classifier | Final Optimized Model |

After evaluation, **LightGBM** achieved the best overall performance and was selected as the final deployment model.

---

# ⚙️ Hyperparameter Tuning

The LightGBM model was optimized using **GridSearchCV**.

Parameters tuned include:

- num_leaves
- learning_rate
- n_estimators
- max_depth
- subsample
- min_child_samples

This optimization improved overall prediction performance.

---

# 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

### Model Comparison

| Metric | Random Forest | LightGBM |
|---------|---------------|-----------|
| Accuracy | Better baseline | ✅ Best |
| Precision | High | ✅ Higher |
| Recall | Good | ✅ Better |
| F1 Score | High | ✅ Highest |
| ROC-AUC | Excellent | ✅ Best |

**Final Selected Model:** **LightGBM**

---

# 🧪 Robustness Validation

To ensure the model generalizes well, robustness testing was performed using:

- K-Fold Cross Validation
- Multiple Random Seeds
- Noise Injection Testing
- Train-Test Split Variations

The final model demonstrated stable performance across different testing scenarios.

---

# 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive Explanations) was used to interpret model predictions.

Key observations:

- Torque is the strongest indicator of machine failure.
- Tool Wear has significant influence.
- Temperature Difference contributes to operational stress.
- Rotational Speed affects failure probability.

This improves model transparency and interpretability.

---

# 🚀 Deployment

The trained LightGBM model was deployed using **Streamlit**.

### Application Features

- Interactive user interface
- Manual sensor input
- Real-time prediction
- Failure probability score
- Clean and responsive dashboard

---

# 📁 Project Structure

```
Predictive-Maintenance-System/
│
├── app.py
├── README.md
│
├── data/
│   ├── ai4i2020.csv
│
├── models/
│   ├── final_lightgbm_model.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│
├── notebooks/
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── smote_pipeline.py
│   ├── hyperparameter_tuning.py
│   ├── shap_analysis.py
│   ├── compare_models.py
│   └── final_evaluation.py
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── shap_feature_importance.png
│   ├── heatmap.png
│
└── requirements.txt
```

---

# 💻 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Random Forest
- SHAP
- SMOTE (imbalanced-learn)
- Matplotlib
- Joblib
- Streamlit

---

# 📌 Future Enhancements

- Real-time IoT sensor integration
- Predictive maintenance dashboard
- Cloud deployment (AWS/Azure/GCP)
- REST API using FastAPI
- Docker containerization
- Automated model retraining pipeline

---

# 👨‍💻 Author

**Sherin John**

Machine Learning | Data Science | Artificial Intelligence

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub!
