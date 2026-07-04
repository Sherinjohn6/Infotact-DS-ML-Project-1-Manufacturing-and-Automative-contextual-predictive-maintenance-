# AI4I Predictive Maintenance using Machine Learning

## Project Overview

This project develops a machine learning model to predict machine failures using the **AI4I 2020 Predictive Maintenance Dataset**. The objective is to identify potential failures before they occur, enabling preventive maintenance, reducing downtime, and improving operational efficiency.

The project follows a complete end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, explainable AI (SHAP), and model deployment preparation.

---

## Objectives

* Predict machine failure accurately.
* Compare multiple machine learning models.
* Optimize model performance through hyperparameter tuning.
* Improve classification using threshold optimization.
* Validate model robustness with cross-validation.
* Explain predictions using SHAP.
* Save the trained model for deployment.

---

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

### Features

* Type
* Air Temperature (K)
* Process Temperature (K)
* Rotational Speed (rpm)
* Torque (Nm)
* Tool Wear (min)
* Engineered Features (created during feature engineering)

### Target Variable

* Machine Failure

  * 0 = No Failure
  * 1 = Failure

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Comparison
7. Hyperparameter Tuning
8. Threshold Optimization
9. Robust Cross Validation
10. Final Model Evaluation
11. SHAP Explainability
12. Save Final Model
13. Deployment Preparation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* LightGBM
* SHAP
* Joblib

---

## Machine Learning Models Evaluated

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

The final selected model is the tuned **LightGBM** classifier because it achieved the best overall performance.

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** to identify the optimal parameter combination for the LightGBM model.

The tuned model improved overall prediction accuracy and generalization performance.

---

## Threshold Optimization

Instead of using the default classification threshold (0.5), threshold optimization was performed to improve the balance between precision and recall.

This resulted in better failure detection performance.

---

## Robust Cross Validation

K-Fold Cross Validation was used to evaluate the stability and robustness of the model.

Metrics evaluated include:

* Accuracy
* Precision
* Recall
* F1-Score

The model demonstrated consistent performance across all folds.

---

## Model Evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report

---

## Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was used to explain the predictions made by the LightGBM model.

Visualizations include:

* SHAP Feature Importance
* SHAP Summary (Beeswarm) Plot

These plots highlight the most influential features affecting machine failure predictions.

---

## Saved Model Files

The following files are included for deployment:

* `final_lightgbm_model.pkl`
* `label_encoder.pkl`
* `feature_names.pkl`

These files can be loaded directly into a deployment application.

---

## Project Structure

```text
AI4I-Predictive-Maintenance/
│
├── dataset/
├── notebooks/
├── models/
│   ├── final_lightgbm_model.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
│
├── save_model.py
├── shap_analysis.py
├── app.py
├── requirements.txt
├── README.md
└── images/
```

---

## Results

* High prediction accuracy using the tuned LightGBM model.
* Reliable performance validated through cross-validation.
* Improved classification using threshold optimization.
* Model predictions explained using SHAP for transparency and interpretability.

---

## Future Improvements

* Deploy the model using Streamlit.
* Integrate real-time sensor data.
* Monitor deployed model performance.
* Explore deep learning approaches.
* Implement cloud deployment.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sherinjohn6/Infotact-DS-ML-Project-1.git
```

Navigate to the project folder:

```bash
cd Infotact-DS-ML-Project-1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Author

**Sherin John**

Machine Learning | Data Science | Predictive Maintenance

---

## License

This project is developed for educational and portfolio purposes.
