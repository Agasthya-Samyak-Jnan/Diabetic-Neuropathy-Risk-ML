import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import confusion_matrix, mean_absolute_error

from model import CNNModel

# Windows to Linux file path conversion
def fix_path(p):
    return os.path.abspath(p.strip().replace("\\", "/"))

# Paths                                              
CSV_FILE    = "Extended_Dataset/labels.csv"  # Make sure Dataset Images are at relative path ./Extended_Dataset/ from this python script's location.
MODEL_PATH  = "MobileNetV2-base-model-v02.keras"
OUTPUT_XGB  = "xgb_cnn_model.joblib"

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(CSV_FILE, header=None, names=["image_path", "label"])
print(f"Found {len(df)} images in CSV")

# -----------------------------
# Initialize Hybrid Model
# -----------------------------
cnn_model = CNNModel(MODEL_PATH)

# -----------------------------
# Convert label strings to numeric (ordinal)
# Example: LOW=0, MEDIUM=1, HIGH=2
# -----------------------------
label_map = {"LOW":0, "MEDIUM":1, "HIGH":2}
df['label_num'] = df['label'].map(label_map)

# -----------------------------
# Split CSV into train/val based on path
# -----------------------------
train_df = df[df['image_path'].str.contains('train', case=False)]
val_df   = df[df['image_path'].str.contains('val', case=False)]

print(f"Train images: {len(train_df)}, Val images: {len(val_df)}")

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features_from_df(df_subset):
    features_list = []
    labels_list = []
    for idx, row in df_subset.iterrows():
        path = fix_path(row['image_path'])
        label = row['label_num']

        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue

        features = cnn_model.predict(path)
        features_list.append(features)
        labels_list.append(label)
    return np.array(features_list), np.array(labels_list, dtype=float)

X_train, y_train = extract_features_from_df(train_df)
X_val, y_val     = extract_features_from_df(val_df)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# -----------------------------
# Train XGBoost Regressor (ordinal regression style)
# -----------------------------
xgb_reg = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

xgb_reg.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluate on validation set
# -----------------------------
y_pred = xgb_reg.predict(X_val_scaled)

# RMSE
rmse = root_mean_squared_error(y_val, y_pred)

# Round predictions (ordinal classification)
y_pred_rounded = np.clip(np.round(y_pred), 0, y_train.max())

# Ordinal Accuracy (±1 tolerance)
ordinal_acc = np.mean(np.abs(y_pred_rounded - y_val) <= 1)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_rounded)

# MAE
mae = mean_absolute_error(y_val, y_pred)

# Prediction distribution
unique, counts = np.unique(y_pred_rounded, return_counts=True)
pred_dist = dict(zip(unique, counts))

print("\n========== VALIDATION RESULTS ==========")
print(f"X_val shape: {X_val.shape}")

print(f"Ordinal accuracy (±1): {ordinal_acc}")
print("Confusion matrix:\n", cm)

print(f"Validation MAE: {mae}")
print(f"Validation RMSE: {rmse:.4f}")

print(pred_dist)
print("========================================\n")

# Optional: round predictions to get discrete labels
y_pred_rounded = np.clip(np.round(y_pred), 0, y_train.max())
print("Rounded predictions:", y_pred_rounded)

# -----------------------------
# Save model & scaler
# -----------------------------
joblib.dump(xgb_reg, OUTPUT_XGB)
joblib.dump(scaler, "scaler.joblib")
print(f"Saved XGBoost model to {OUTPUT_XGB} and scaler to scaler.joblib")
