# ==========================================
# Save Final Tuned LightGBM Model
# AI4I Predictive Maintenance
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("project1_feature_engineered.csv")

# Encode categorical column
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# Features and Target
X = df.drop(["Machine failure", "UDI", "Product ID"], axis=1)
y = df["Machine failure"]

# Clean column names
X.columns = (
    X.columns
    .str.replace(" ", "_", regex=False)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the trained model
model = joblib.load("lightgbm_tuned_model.pkl")

# Save the final model
joblib.dump(model, "final_lightgbm_model.pkl")

# Save Label Encoder
joblib.dump(le, "label_encoder.pkl")

# Save Feature Names
joblib.dump(list(X.columns), "feature_names.pkl")

print("✅ Final model saved successfully!")
print("Model: final_lightgbm_model.pkl")
print("Label Encoder: label_encoder.pkl")
print("Feature Names: feature_names.pkl")