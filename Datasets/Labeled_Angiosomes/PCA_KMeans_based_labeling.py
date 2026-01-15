# =========================================================
# FULLY UNSUPERVISED DIABETIC NEUROPATHY RISK LABELER
# IEEE THERMOGRAPHY DATASET (CSV-BASED)
# =========================================================

import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ================= USER SETTINGS =================
DATA_ROOT = "ThermoDataBase_IEEE_DataPort"
N_CLUSTERS = 3
OUTPUT_FILE = "unsupervised_neuropathy_risk_labels.csv"
# ================================================


# ------------------------------------------------
# 1. FIND ALL CSV FILES
# ------------------------------------------------
csv_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True)

if len(csv_files) == 0:
    raise RuntimeError("❌ No CSV files found. Check DATA_ROOT path.")

print(f"✅ Found {len(csv_files)} CSV files")


# ------------------------------------------------
# 2. FEATURE EXTRACTION (ROBUST TO SHAPE)
# ------------------------------------------------
def extract_features(csv_path):
    df = pd.read_csv(csv_path)

    # Flatten all temperature values
    vals = df.values.flatten()
    vals = vals[~np.isnan(vals)]

    if len(vals) < 10:  # safety check
        return None

    mean_temp = np.mean(vals)
    std_temp  = np.std(vals)
    range_temp = np.max(vals) - np.min(vals)
    p95_temp = np.percentile(vals, 95)

    # Thermal entropy (irregularity)
    hist, _ = np.histogram(vals, bins=64, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))

    return [
        mean_temp,
        std_temp,
        range_temp,
        entropy,
        p95_temp
    ]


# ------------------------------------------------
# 3. BUILD FEATURE MATRIX
# ------------------------------------------------
X = []
file_ids = []

for csv in csv_files:
    feats = extract_features(csv)
    if feats is not None:
        X.append(feats)
        file_ids.append(os.path.relpath(csv, DATA_ROOT))

X = np.array(X)
print("✅ Feature matrix shape:", X.shape)

# ------------------------------------------------
# 4. NORMALIZATION
# ------------------------------------------------
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

# ------------------------------------------------
# 5. PCA-BASED THERMAL RISK SCORE (DATA-DRIVEN)
# ------------------------------------------------
pca = PCA(n_components=1)
TRS = pca.fit_transform(Xn).flatten()

# Normalize TRS to [0, 1]
TRS = (TRS - TRS.min()) / (TRS.max() - TRS.min())

# ------------------------------------------------
# 6. UNSUPERVISED CLUSTERING
# ------------------------------------------------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
clusters = kmeans.fit_predict(Xn)

# ------------------------------------------------
# 7. AUTOMATIC RISK LABEL ASSIGNMENT
# ------------------------------------------------
cluster_trs_mean = {
    c: TRS[clusters == c].mean()
    for c in range(N_CLUSTERS)
}

sorted_clusters = sorted(cluster_trs_mean, key=cluster_trs_mean.get)

risk_map = {
    sorted_clusters[0]: "LOW",
    sorted_clusters[1]: "MEDIUM",
    sorted_clusters[2]: "HIGH"
}

risk_labels = [risk_map[c] for c in clusters]

# ------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------
out = pd.DataFrame({
    "sample": file_ids,
    "cluster": clusters,
    "TRS": TRS,
    "risk_label": risk_labels
})

out.to_csv(OUTPUT_FILE, index=False)

print("\n✅ DONE")
print("Risk distribution:")
print(out["risk_label"].value_counts())
print(f"\n📁 Saved to: {OUTPUT_FILE}")
