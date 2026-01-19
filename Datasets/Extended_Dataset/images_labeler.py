import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import entropy

# =====================================================
# 1. Thermal feature extraction (INAOEE color thermograms)
# =====================================================
def extract_thermal_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Use V channel as temperature proxy
    temp = hsv[:, :, 2].astype(np.float32)

    # ---- Global thermal features (same as CSV pipeline) ----
    mean_temp = np.mean(temp)
    std_temp = np.std(temp)

    hist = np.histogram(temp, bins=256, range=(0, 255), density=True)[0]
    ent = entropy(hist + 1e-6)

    h, w = temp.shape
    left = temp[:, :w // 2]
    right = temp[:, w // 2:]
    asym = abs(np.mean(left) - np.mean(right))

    # Thermal Risk Score (same weights)
    trs = 0.4 * mean_temp + 0.3 * std_temp + 0.3 * ent

    return mean_temp, std_temp, ent, asym, trs

# =====================================================
# 2. Load all PNG images recursively
# =====================================================
DATASET_ROOT = "Extended_Dataset"  # CHANGE THIS

records = []

for root, _, files in os.walk(DATASET_ROOT):
    for file in files:
        if file.lower().endswith(".png"):
            path = os.path.join(root, file)
            feats = extract_thermal_features(path)
            if feats is None:
                continue

            records.append({
                "image_path": path,
                "mean_temp": feats[0],
                "std_temp": feats[1],
                "entropy": feats[2],
                "asymmetry": feats[3],
                "TRS": feats[4]
            })

df = pd.DataFrame(records)
print(f"Found {len(df)} PNG images")

# =====================================================
# 3. Normalize + KMeans clustering
# =====================================================
X = df[["mean_temp", "std_temp", "entropy", "asymmetry", "TRS"]].values
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X)

# =====================================================
# 4. Cluster -> LOW / MEDIUM / HIGH
# =====================================================
cluster_order = (
    df.groupby("cluster")["TRS"]
      .mean()
      .sort_values()
      .index
      .tolist()
)

label_map = {
    cluster_order[0]: "LOW",
    cluster_order[1]: "MEDIUM",
    cluster_order[2]: "HIGH"
}

df["risk_label"] = df["cluster"].map(label_map)

# =====================================================
# 5. Save output
# =====================================================
OUTPUT_FILE = "unsupervised_angiosome_png_risk_labels.csv"
df[["image_path", "risk_label"]].to_csv(OUTPUT_FILE, index=False)

print("\n DONE")
print(df["risk_label"].value_counts())
