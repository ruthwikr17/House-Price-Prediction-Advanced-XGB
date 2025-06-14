import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Set paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "../data/Indian_Real_Estate_Clean_Data.csv"
)

# Load XGBoost model and full preprocessor pipeline
xgb_model = joblib.load(XGB_MODEL_PATH)

# Load and prepare dataset
df = pd.read_csv(DATA_PATH)
selected_features = [
    "Location",
    "Total_Area(SQFT)",
    "Total_Rooms",
    "Balcony",
    "city",
    "property_type",
    "BHK",
]
df = df[selected_features + ["Price"]].dropna()
X = df[selected_features].copy()
y = df["Price"]

target_encoder, numeric_imputer = joblib.load(PREPROCESSOR_PATH)

categorical_features = ["Location", "Balcony", "city", "property_type"]
numeric_features = ["Total_Area(SQFT)", "Total_Rooms", "BHK"]

# 1. Apply target encoding
X[categorical_features] = target_encoder.transform(X[categorical_features])

# Transform using NumPy arrays to avoid feature name mismatch issues
X_imputed = numeric_imputer.transform(X.values)
X = pd.DataFrame(X_imputed, columns=X.columns)

# 2. Reorder the columns exactly as in training
all_features_ordered = numeric_features + categorical_features
X = X[all_features_ordered]

X_processed = X

# Optional: convert to DataFrame for better SHAP visualization
if not isinstance(X_processed, pd.DataFrame):
    X_processed = pd.DataFrame(X_processed)

# ---- SHAP EXPLAINABILITY ---- #
print("Generating SHAP values for XGBoost...")
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X)

# 1. SHAP Summary Plot
shap.summary_plot(shap_values, X_processed, show=True)

# 2. SHAP Bar Plot (Top 10)
shap.plots.bar(shap_values, max_display=10)

# 3. Local Explanation (first sample)
sample_idx = 0
shap.plots.waterfall(shap_values[sample_idx], max_display=10)

# Optional: Save summary plot as PNG
# plt.savefig("shap_summary.png", bbox_inches="tight")
