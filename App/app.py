import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import joblib

from model import HybridModel

# -----------------------------
# Load models
# -----------------------------
MODEL_PATH = "models/MobileNetV2-base-model-v02.keras"
XGB_PATH   = "models/xgb_hybrid_model.joblib"
SCALER_PATH = "models/scaler.joblib"

hybrid_model = HybridModel(MODEL_PATH)
xgb_model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Risk mapping
# -----------------------------
def risk_label(val):
    if val <= 0.5:
        return "LOW RISK"
    elif val <= 1.5:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

# -----------------------------
# App window
# -----------------------------
root = tk.Tk()
root.title("Diabetic Neuropathy Risk Prediction")
root.geometry("600x720")
root.resizable(False, False)

selected_image_path = None

# -----------------------------
# Functions
# -----------------------------
def upload_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(
        title="Select Thermal Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    if file_path:
        selected_image_path = file_path

        img = Image.open(file_path)
        img = img.resize((350, 350))
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="Image loaded. Ready to predict.")

def predict_risk():
    if selected_image_path is None:
        messagebox.showwarning("No image", "Please upload an image first.")
        return

    try:
        features = hybrid_model.predict(selected_image_path)
        features = scaler.transform([features])

        pred = xgb_model.predict(features)[0]
        pred_rounded = int(np.clip(round(pred), 0, 2))

        result_label.config(
            text=f"Predicted Risk: {risk_label(pred)}\n(Score: {pred:.2f})",
            fg="blue"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# -----------------------------
# UI Elements
# -----------------------------
title = tk.Label(
    root,
    text="🦶 Diabetic Neuropathy Risk Predictor",
    font=("Arial", 16, "bold")
)
title.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

upload_btn = tk.Button(
    root,
    text="Upload Thermal Image",
    command=upload_image,
    width=25,
    height=2
)
upload_btn.pack(pady=10)

predict_btn = tk.Button(
    root,
    text="Predict Risk",
    command=predict_risk,
    width=25,
    height=2,
    bg="#4CAF50",
    fg="white"
)
predict_btn.pack(pady=10)

result_label = tk.Label(
    root,
    text="Upload an image to begin.",
    font=("Arial", 12)
)
result_label.pack(pady=20)

# -----------------------------
# Run app
# -----------------------------
root.mainloop()
